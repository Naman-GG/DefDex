"""
DefDex — Stage 7: Capability-asymmetry win-probability model (LogReg + MLP)
===========================================================================

Produces a calibrated P(country A defeats country B) from the two countries'
capability vectors. This is a *capability-advantage* model, NOT a forecast of
real war outcomes — an earlier experiment training on real historical outcomes
(Correlates of War) confirmed that current capability differences do not
predict who actually wins wars (CV AUC ~ 0.5; terrain, strategy, alliances and
resolve dominate). So we model the well-posed question we *can* answer:
"given the force balance, how large is the measured capability advantage,
expressed as a calibrated probability?"

Method (fully transparent):
  1. Combat Power (CP): a doctrine-weighted blend of the six domain indices,
     normalized to 0-1. Weights are explicit and documented below.
  2. Simulated matchups: for many ordered country pairs we draw a win/loss from
     Bernoulli(sigmoid(K * (CP_A - CP_B))). The logistic + sampling injects
     realistic upsets (close matchups are near coin-flips; lopsided ones aren't).
  3. Models: Logistic Regression (interpretable domain weights) and an MLP
     (nonlinear), ensembled by averaging probabilities. They learn to map the
     six per-domain capability *differences* to a calibrated win probability.

Inputs:
    data/processed/features.csv
Outputs:
    model/win_predictor.pkl
    data/processed/win_probabilities.csv   (focus matchups + full matrix long-form)

Run:
    milenv/bin/python model/train_win_predictor.py
"""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "model"
RANDOM_STATE = 42

# Six domain-strength features (higher = stronger). import_dependency is a
# vulnerability, so it enters Combat Power with a negative weight.
DOMAIN_FEATS = [
    "weaponry_index", "manpower_index", "economic_resilience_index",
    "conflict_experience_index", "defense_industry_index", "import_dependency_index",
]
# Doctrine-based Combat-Power weights (documented, sum of |w| ~ 1).
CP_WEIGHTS = {
    "weaponry_index": 0.32,            # hardware decisive in conventional war
    "economic_resilience_index": 0.22, # sustainment / industrial base
    "manpower_index": 0.16,
    "defense_industry_index": 0.16,    # self-sufficiency under attrition
    "conflict_experience_index": 0.14, # battle-tested
    "import_dependency_index": -0.12,  # supply vulnerability (penalty)
}
K_STEEP = 6.0  # logistic steepness: CP gap of ~0.18 -> ~75% win probability
N_REPLICATES = 60  # Bernoulli draws per ordered pair


def combat_power(cap: pd.DataFrame) -> pd.Series:
    cp = sum(cap[f] * w for f, w in CP_WEIGHTS.items())
    cp = (cp - cp.min()) / (cp.max() - cp.min())  # normalize to 0-1
    return cp.round(4)


def load_cap() -> tuple[pd.DataFrame, pd.Series]:
    f = pd.read_csv(PROCESSED / "features.csv").set_index("country")
    cap = f[DOMAIN_FEATS].copy()
    cap = cap.fillna(cap.median(numeric_only=True))  # impute Taiwan/N.Korea gaps
    return cap, combat_power(cap)


def simulate(cap: pd.DataFrame, cp: pd.Series, rng: np.random.Generator) -> pd.DataFrame:
    """Build simulated matchup rows: features = domain diffs, label ~ Bernoulli."""
    countries = list(cap.index)
    rows_feat, labels = [], []
    for a in countries:
        for b in countries:
            if a == b:
                continue
            diff = (cap.loc[a] - cap.loc[b]).values
            p = 1 / (1 + np.exp(-K_STEEP * (cp[a] - cp[b])))
            draws = rng.binomial(1, p, size=N_REPLICATES)
            rows_feat.extend([diff] * N_REPLICATES)
            labels.extend(draws.tolist())
    X = pd.DataFrame(rows_feat, columns=[f"d_{c}" for c in DOMAIN_FEATS])
    return X.assign(y=labels)


def make_models() -> tuple[Pipeline, Pipeline]:
    logreg = Pipeline([("scale", StandardScaler()),
                       ("clf", LogisticRegression(C=1.0, max_iter=2000, random_state=RANDOM_STATE))])
    mlp = Pipeline([("scale", StandardScaler()),
                    ("clf", MLPClassifier(hidden_layer_sizes=(16, 8), alpha=0.5,
                                          max_iter=1500, random_state=RANDOM_STATE))])
    return logreg, mlp


def win_prob(logreg, mlp, cap, a, b) -> float:
    diff = (cap.loc[a] - cap.loc[b]).values.reshape(1, -1)
    row = pd.DataFrame(diff, columns=[f"d_{c}" for c in DOMAIN_FEATS])
    return float((logreg.predict_proba(row)[:, 1] + mlp.predict_proba(row)[:, 1])[0] / 2)


def main() -> None:
    rng = np.random.default_rng(RANDOM_STATE)
    cap, cp = load_cap()
    print(f"[DefDex] Win predictor: {len(cap)} countries; Combat Power range "
          f"{cp.min():.2f}-{cp.max():.2f}")

    data = simulate(cap, cp, rng)
    X, y = data[[f"d_{c}" for c in DOMAIN_FEATS]], data["y"]

    # Hold out a random split to check the fit recovers calibrated structure.
    n = len(X)
    idx = rng.permutation(n)
    cut = int(0.8 * n)
    tr, te = idx[:cut], idx[cut:]
    logreg, mlp = make_models()
    logreg.fit(X.iloc[tr], y.iloc[tr])
    mlp.fit(X.iloc[tr], y.iloc[tr])
    p_te = 0.5 * (logreg.predict_proba(X.iloc[te])[:, 1] + mlp.predict_proba(X.iloc[te])[:, 1])
    print(f"[eval] held-out ensemble AUC vs sampled labels = {roc_auc_score(y.iloc[te], p_te):.3f}")

    # Refit on all simulated data for deployment.
    logreg.fit(X, y)
    mlp.fit(X, y)

    coefs = pd.Series(logreg.named_steps["clf"].coef_[0], index=DOMAIN_FEATS).sort_values(key=abs, ascending=False)
    print("\nLogReg standardized coefficients (domain drivers of win probability):")
    print(coefs.round(3).to_string())

    # ---- Focus matchups ----
    focus_pairs = [("India", "China"), ("India", "Pakistan"), ("China", "Pakistan"),
                   ("China", "India"), ("Pakistan", "India"), ("Pakistan", "China")]
    focus = pd.DataFrame([
        {"side_a": a, "side_b": b, "cp_a": cp[a], "cp_b": cp[b],
         "win_prob_a": round(win_prob(logreg, mlp, cap, a, b), 3)}
        for a, b in focus_pairs
    ])
    print("\nFocus matchups — P(side_a defeats side_b):")
    print(focus.to_string(index=False))

    # ---- Full long-form matrix for all 50x50 (for dashboard) ----
    full = pd.DataFrame([
        {"side_a": a, "side_b": b, "win_prob_a": round(win_prob(logreg, mlp, cap, a, b), 3)}
        for a in cap.index for b in cap.index if a != b
    ])
    full.to_csv(PROCESSED / "win_probabilities.csv", index=False)

    # ---- India-centric win-probability plot ----
    india_row = full[full.side_a == "India"].set_index("side_b")["win_prob_a"]
    targets = ["China", "Pakistan"]
    vals = [india_row[t] for t in targets]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar([f"India vs\n{t}" for t in targets], vals,
                  color=["#c0392b", "#27ae60"])
    ax.axhline(0.5, color="gray", ls="--", lw=1, label="even odds")
    ax.set_ylim(0, 1)
    ax.set_ylabel("P(India wins)")
    ax.set_title("Capability-advantage win probability — India")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.0%}", ha="center")
    ax.legend()
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "win_probability_india.png", dpi=130)
    plt.close()

    joblib.dump(
        {"logreg": logreg, "mlp": mlp, "domain_feats": DOMAIN_FEATS,
         "combat_power": cp.to_dict(), "cp_weights": CP_WEIGHTS, "k_steep": K_STEEP},
        MODEL_DIR / "win_predictor.pkl",
    )
    print(f"\n[DefDex] Wrote model/win_predictor.pkl and "
          f"data/processed/win_probabilities.csv ({len(full)} matchups)")


if __name__ == "__main__":
    main()

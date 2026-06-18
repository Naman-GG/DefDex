"""
DefDex — Stage 6: XGBoost Capability Scorer + SHAP explainability
=================================================================

Trains a gradient-boosted regressor to predict a country's military capability
(`strength_score`, 0-1, higher = stronger) from its granular feature vector,
then uses SHAP to explain *what drives* each score — both at the granular
feature level and aggregated to the six capability domains.

Note on circularity: the target derives from GFP's Power Index, which GFP
computes from many of these same inputs (budget, hardware, GDP). So the model
is approximating GFP's undisclosed weighting; SHAP is what makes that
approximation useful — it surfaces the implicit importance of each driver and
gives per-country, per-domain attributions that feed the Stage-8 gap analyzer.

Inputs:
    data/processed/features.csv
    data/processed/feature_dictionary.csv   (feature -> domain map)

Outputs:
    model/capability_scorer.pkl              trained model + metadata
    data/processed/capability_scores.csv     actual/predicted + per-domain SHAP
    model/shap_summary.png                   global SHAP beeswarm
    model/shap_domain_contributions.png      domain SHAP for India/China/Pakistan

Run:
    milenv/bin/python model/train_capability_scorer.py
"""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import KFold, cross_val_predict
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "model"

TARGET = "strength_score"
DROP = {"country", "power_index", "strength_score"}
FOCUS = ["India", "China", "Pakistan"]
RANDOM_STATE = 42


def load_xy() -> tuple[pd.DataFrame, pd.Series, pd.Series, dict[str, str]]:
    df = pd.read_csv(PROCESSED / "features.csv")
    dct = pd.read_csv(PROCESSED / "feature_dictionary.csv")

    # Granular features only: drop targets and the composite *_index aggregates
    # (they're built from the raw features and would mask granular drivers).
    feat_cols = [
        c for c in df.columns
        if c not in DROP and not c.endswith("_index")
    ]
    X = df[feat_cols].copy()
    y = df[TARGET].copy()

    # feature -> domain (granular features only)
    domain_map = dict(zip(dct["feature"], dct["domain"]))
    domain_map = {f: domain_map[f] for f in feat_cols if f in domain_map}
    return X, y, df["country"], domain_map


def build_model() -> XGBRegressor:
    # Small, regularized: 50 rows x ~37 features -> guard against overfitting.
    return XGBRegressor(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.8,
        reg_lambda=1.5,
        min_child_weight=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def evaluate(model: XGBRegressor, X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """5-fold CV; return out-of-fold predictions and print metrics."""
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof = cross_val_predict(model, X, y, cv=kf)
    resid = y - oof
    ss_res = float((resid**2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot
    rmse = float(np.sqrt((resid**2).mean()))
    mae = float(resid.abs().mean())
    print(f"[CV] 5-fold out-of-fold:  R2={r2:.3f}  RMSE={rmse:.4f}  MAE={mae:.4f}")
    return pd.Series(oof, index=X.index, name="cv_predicted")


def main() -> None:
    X, y, countries, domain_map = load_xy()
    print(f"[DefDex] Capability scorer: {X.shape[0]} countries x {X.shape[1]} features")

    model = build_model()
    oof = evaluate(model, X, y)

    # Final fit on all rows for the deployed scorer + SHAP.
    model.fit(X, y)

    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)  # (n_countries, n_features)
    base_value = float(explainer.expected_value)

    # ---- Global feature importance (mean |SHAP|) ----
    importance = (
        pd.Series(np.abs(sv).mean(axis=0), index=X.columns)
        .sort_values(ascending=False)
    )
    print("\nTop 10 global drivers (mean |SHAP|):")
    print(importance.head(10).round(4).to_string())

    # ---- Domain-aggregated SHAP (signed) per country ----
    domains = sorted(set(domain_map.values()))
    shap_df = pd.DataFrame(sv, columns=X.columns, index=countries)
    dom_contrib = pd.DataFrame(index=countries)
    for d in domains:
        cols = [c for c in X.columns if domain_map.get(c) == d]
        dom_contrib[f"shap_{d}"] = shap_df[cols].sum(axis=1)

    # ---- Assemble scores table ----
    scores = pd.DataFrame({
        "country": countries,
        "strength_score": y.values,
        "predicted": model.predict(X),
        "cv_predicted": oof.values,
        "shap_base_value": base_value,
    })
    scores = pd.concat([scores.set_index("country"), dom_contrib], axis=1).reset_index()
    scores["residual"] = (scores["strength_score"] - scores["predicted"]).round(4)
    for c in scores.select_dtypes("number").columns:
        scores[c] = scores[c].round(4)
    scores = scores.sort_values("strength_score", ascending=False).reset_index(drop=True)
    out_csv = PROCESSED / "capability_scores.csv"
    scores.to_csv(out_csv, index=False)

    print(f"\nDomain SHAP contributions (signed, vs base={base_value:.3f}):")
    show = scores[scores.country.isin(FOCUS)].set_index("country")
    print(show[["strength_score", "predicted",
                *[f"shap_{d}" for d in domains]]].to_string())

    # ---- Persist model ----
    joblib.dump(
        {"model": model, "features": list(X.columns), "domain_map": domain_map,
         "target": TARGET, "base_value": base_value},
        MODEL_DIR / "capability_scorer.pkl",
    )

    # ---- Plots ----
    shap.summary_plot(sv, X, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "shap_summary.png", dpi=130, bbox_inches="tight")
    plt.close()

    ax = show[[f"shap_{d}" for d in domains]].rename(
        columns=lambda c: c.replace("shap_", "")
    ).T.plot(kind="bar", figsize=(11, 6))
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("SHAP contribution to strength_score")
    ax.set_title("Capability drivers by domain — India vs China vs Pakistan")
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "shap_domain_contributions.png", dpi=130)
    plt.close()

    print(f"\n[DefDex] Wrote {out_csv.relative_to(ROOT)}")
    print(f"[DefDex] Wrote {(MODEL_DIR / 'capability_scorer.pkl').relative_to(ROOT)}")
    print(f"[DefDex] Wrote model/shap_summary.png, model/shap_domain_contributions.png")


if __name__ == "__main__":
    main()

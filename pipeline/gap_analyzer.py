"""
DefDex — Stage 8: Gap Analyzer & Recommendations Engine
=======================================================

Two analyses built on the Stage-5 feature matrix:

  1. Capability clustering — KMeans groups the 50 countries by their six-domain
     capability profile (k chosen by silhouette score). This locates India's
     peer tier and who sits above it.

  2. Gap analysis + recommendations — quantifies India's per-domain shortfall
     against China (its aspirational peer / primary adversary), drills into the
     specific granular features driving each domain gap, and emits a ranked,
     data-driven recommendation list.

Inputs:
    data/processed/features.csv
    data/processed/feature_dictionary.csv
Outputs:
    data/processed/country_clusters.csv
    data/processed/india_gap_analysis.csv
    data/processed/recommendations.csv
    model/cluster_map.png          PCA scatter of capability clusters
    model/india_gap.png            India vs China per-domain gap

Run:
    milenv/bin/python pipeline/gap_analyzer.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "model"
RANDOM_STATE = 42

BENCHMARK = "China"   # India's aspirational peer for gap analysis
SUBJECT = "India"

DOMAIN_FEATS = [
    "weaponry_index", "manpower_index", "economic_resilience_index",
    "conflict_experience_index", "defense_industry_index", "import_dependency_index",
]
# Domains where a higher value is BETTER for the subject. import_dependency is a
# vulnerability (higher = worse), handled with sign flips below.
HIGHER_BETTER = {f: True for f in DOMAIN_FEATS}
HIGHER_BETTER["import_dependency_index"] = False

# Actionable recommendation templates per domain (conflict_experience excluded —
# not a policy lever).
REC_TEMPLATES = {
    "weaponry_index": "Modernize and expand the force — prioritize the platforms below where India trails most.",
    "economic_resilience_index": "Broaden the economic/industrial base funding defense (GDP growth, budget, energy reserves).",
    "defense_industry_index": "Scale domestic defense manufacturing and arms exports to build an indigenous industry.",
    "import_dependency_index": "Cut reliance on foreign arms suppliers by substituting imports with domestic production.",
    "manpower_index": "Sustain personnel readiness and reserve depth.",
}


def load() -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(PROCESSED / "features.csv").set_index("country")
    dct = pd.read_csv(PROCESSED / "feature_dictionary.csv")
    domain_of = dict(zip(dct["feature"], dct["domain"]))
    return df, domain_of


# --------------------------------------------------------------------------- #
# 1. Clustering
# --------------------------------------------------------------------------- #
def cluster(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, StandardScaler, KMeans]:
    X = df[DOMAIN_FEATS].copy()
    X = X.fillna(X.median(numeric_only=True))  # impute Taiwan/N.Korea gaps
    Xs = StandardScaler().fit_transform(X)

    # Search k>=3: k=2 trivially isolates the USA outlier and yields no useful
    # tiering, so we require at least three capability strata.
    best_k, best_score = 3, -1.0
    for k in range(3, 9):
        labels = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit_predict(Xs)
        score = silhouette_score(Xs, labels)
        if score > best_score:
            best_k, best_score = k, score
    print(f"[cluster] best k={best_k} (silhouette={best_score:.3f})")

    km = KMeans(n_clusters=best_k, n_init=10, random_state=RANDOM_STATE).fit(Xs)
    out = df[["strength_score", *DOMAIN_FEATS]].copy()
    out["cluster"] = km.labels_

    # Label clusters as tiers by descending mean strength.
    order = out.groupby("cluster")["strength_score"].mean().sort_values(ascending=False)
    tier = {c: f"Tier {i+1}" for i, c in enumerate(order.index)}
    out["tier"] = out["cluster"].map(tier)
    return out, Xs, km, order


# --------------------------------------------------------------------------- #
# 2. Gap analysis + recommendations
# --------------------------------------------------------------------------- #
def gap_analysis(df: pd.DataFrame, domain_of: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    subj, bench = df.loc[SUBJECT], df.loc[BENCHMARK]

    rows = []
    for f in DOMAIN_FEATS:
        s, b = subj[f], bench[f]
        # Disadvantage = how much worse the subject is than the benchmark.
        disadvantage = (b - s) if HIGHER_BETTER[f] else (s - b)
        rows.append({"domain": f.replace("_index", ""), "feature": f,
                     "india": round(s, 4), "china": round(b, 4),
                     "disadvantage": round(disadvantage, 4)})
    gap = pd.DataFrame(rows).sort_values("disadvantage", ascending=False).reset_index(drop=True)

    # Drill into granular features for each weak domain (positive disadvantage).
    recs = []
    rank = 1
    for _, r in gap[gap["disadvantage"] > 0.02].iterrows():
        feature = r["feature"]
        specifics = _drill(df, feature, domain_of)
        recs.append({
            "rank": rank, "domain": r["domain"], "disadvantage": r["disadvantage"],
            "recommendation": REC_TEMPLATES.get(feature, "Address domain gap."),
            "specific_targets": specifics,
        })
        rank += 1
    return gap, pd.DataFrame(recs)


def _drill(df: pd.DataFrame, feature: str, domain_of: dict) -> str:
    """Specific granular targets where India trails China within a weak domain."""
    # Vulnerability features map to a single arms-trade indicator (sign-aware).
    if feature == "import_dependency_index":
        return _pair(df, "arms_imports_tiv", lower_better=True)
    if feature == "defense_industry_index":
        return _pair(df, "arms_exports_tiv")

    # Higher-is-better domains: rank granular deficits across the 50-country field.
    dict_domain = domain_of.get(feature)  # e.g. weaponry_index -> "weaponry"
    gran = [c for c, d in domain_of.items()
            if d == dict_domain and not c.endswith("_index") and c in df.columns]
    # Drop ratio/strain features that don't translate to "build more".
    gran = [c for c in gran if c not in
            ("defense_burden", "military_spend_pct_gdp", "combat_aircraft_ratio",
             "reserve_ratio", "mil_participation_rate", "coastline_ratio")]
    if not gran:
        return ""
    sub_n = _normalize(df[gran])
    deficit = (sub_n.loc[BENCHMARK] - sub_n.loc[SUBJECT]).sort_values(ascending=False)
    top = [g for g in deficit.index if deficit[g] > 0.05][:3]
    return "; ".join(_pair(df, g) for g in top)


def _pair(df: pd.DataFrame, col: str, lower_better: bool = False) -> str:
    note = " (India higher = more dependent)" if lower_better else ""
    return f"{col}: India {_fmt(df.loc[SUBJECT, col])} vs China {_fmt(df.loc[BENCHMARK, col])}{note}"


def _normalize(d: pd.DataFrame) -> pd.DataFrame:
    return (d - d.min()) / (d.max() - d.min()).replace(0, np.nan)


def _fmt(v: float) -> str:
    if pd.isna(v):
        return "n/a"
    if abs(v) >= 1e9:
        return f"{v/1e9:.0f}B"
    if abs(v) >= 1e6:
        return f"{v/1e6:.1f}M"
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    return f"{v:.2f}"


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_clusters(clusters: pd.DataFrame, Xs: np.ndarray) -> None:
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    xy = pca.fit_transform(Xs)
    fig, ax = plt.subplots(figsize=(11, 7))
    for tier in sorted(clusters["tier"].unique()):
        m = (clusters["tier"] == tier).values
        ax.scatter(xy[m, 0], xy[m, 1], label=tier, s=60, alpha=0.75)
    for name in [SUBJECT, BENCHMARK, "Pakistan", "United States"]:
        i = clusters.index.get_loc(name)
        ax.annotate(name, (xy[i, 0], xy[i, 1]), fontsize=9, fontweight="bold",
                    xytext=(5, 4), textcoords="offset points")
    ax.scatter(xy[clusters.index.get_loc(SUBJECT), 0],
               xy[clusters.index.get_loc(SUBJECT), 1],
               s=240, facecolors="none", edgecolors="red", linewidths=2)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%} var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%} var)")
    ax.set_title("Military capability clusters (50 countries) — India circled")
    ax.legend(title="Capability tier")
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "cluster_map.png", dpi=130)
    plt.close()


def plot_gap(gap: pd.DataFrame) -> None:
    g = gap.sort_values("disadvantage")
    colors = ["#27ae60" if v <= 0 else "#c0392b" for v in g["disadvantage"]]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(g["domain"], g["disadvantage"], color=colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("India's disadvantage vs China  (right = India behind, left = India ahead)")
    ax.set_title("India vs China — capability gap by domain")
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "india_gap.png", dpi=130)
    plt.close()


def main() -> None:
    df, domain_of = load()

    clusters, Xs, km, order = cluster(df)
    clusters.reset_index().to_csv(PROCESSED / "country_clusters.csv", index=False)
    india_tier = clusters.loc[SUBJECT, "tier"]
    peers = [c for c in clusters.index[clusters["tier"] == india_tier] if c != SUBJECT]
    print(f"[cluster] India is in {india_tier} with peers: {', '.join(peers)}")
    above = clusters.index[clusters["tier"] < india_tier].tolist()
    print(f"[cluster] Tiers above India: {', '.join(sorted(set(clusters.loc[above,'tier'])))} "
          f"-> {', '.join(above)}")

    gap, recs = gap_analysis(df, domain_of)
    gap.to_csv(PROCESSED / "india_gap_analysis.csv", index=False)
    recs.to_csv(PROCESSED / "recommendations.csv", index=False)

    print(f"\nIndia vs China — domain gaps (positive = India behind):")
    print(gap[["domain", "india", "china", "disadvantage"]].to_string(index=False))

    print(f"\nRanked recommendations for India:")
    for _, r in recs.iterrows():
        print(f"  {r['rank']}. [{r['domain']}, gap={r['disadvantage']:.3f}] {r['recommendation']}")
        if r["specific_targets"]:
            print(f"      targets: {r['specific_targets']}")

    plot_clusters(clusters, Xs)
    plot_gap(gap)
    print(f"\n[DefDex] Wrote country_clusters.csv, india_gap_analysis.csv, recommendations.csv")
    print(f"[DefDex] Wrote model/cluster_map.png, model/india_gap.png")


if __name__ == "__main__":
    main()

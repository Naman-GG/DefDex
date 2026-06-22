"""
DefDex — Stage 9: Streamlit Dashboard
=====================================

Interactive military-capability comparison dashboard. Leads with the India vs
China / Pakistan story but lets the user compare any two of the 50 countries.

Sections (sidebar):
    Overview            headline scores, tier, win-probability snapshot
    Capability Radar    six-domain profile, A vs B
    Win Probability     live LogReg + MLP inference for the selected pair
    Gap & Recommendations  per-domain gap + ranked, data-grounded actions
    Capability Clusters interactive PCA map of the KMeans tiers

Design notes:
- Capability scores / SHAP / clusters / recommendations are read from the
  precomputed Stage 6-8 CSVs (deterministic for all 50 countries).
- Win probability is computed LIVE from model/win_predictor.pkl (sklearn
  LogReg + MLP) so any A-vs-B pair works — and there is no xgboost/libomp
  runtime dependency, keeping Streamlit Cloud deploys simple.

Run:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "model"

DOMAIN_LABELS = {
    "weaponry_index": "Weaponry",
    "manpower_index": "Manpower",
    "economic_resilience_index": "Economic Resilience",
    "conflict_experience_index": "Conflict Experience",
    "defense_industry_index": "Defense Industry",
    "import_dependency_index": "Import Dependency",
}
# Radar axes: all "higher = stronger". Import dependency is a vulnerability, so
# it is shown inverted as "Supply Security" = 1 - import_dependency.
RADAR_AXES = ["Weaponry", "Manpower", "Economic Resilience",
              "Conflict Experience", "Defense Industry", "Supply Security"]

st.set_page_config(page_title="DefDex 🛡️", page_icon="🛡️", layout="wide")


# --------------------------------------------------------------------------- #
# Cached loaders
# --------------------------------------------------------------------------- #
@st.cache_data
def load_features() -> pd.DataFrame:
    return pd.read_csv(PROCESSED / "features.csv").set_index("country")


@st.cache_data
def load_scores() -> pd.DataFrame:
    return pd.read_csv(PROCESSED / "capability_scores.csv").set_index("country")


@st.cache_data
def load_clusters() -> pd.DataFrame:
    return pd.read_csv(PROCESSED / "country_clusters.csv").set_index("country")


@st.cache_data
def load_recommendations() -> pd.DataFrame:
    return pd.read_csv(PROCESSED / "recommendations.csv")


@st.cache_resource
def load_win_model() -> dict:
    return joblib.load(MODEL_DIR / "win_predictor.pkl")


def radar_values(feats: pd.DataFrame, country: str) -> list[float]:
    r = feats.loc[country]
    return [
        r["weaponry_index"], r["manpower_index"], r["economic_resilience_index"],
        r["conflict_experience_index"], r["defense_industry_index"],
        1 - r["import_dependency_index"],  # supply security
    ]


def win_probability(model: dict, feats: pd.DataFrame, a: str, b: str) -> dict:
    dom = model["domain_feats"]
    cap = feats[dom].fillna(feats[dom].median(numeric_only=True))
    diff = (cap.loc[a] - cap.loc[b]).values.reshape(1, -1)
    row = pd.DataFrame(diff, columns=[f"d_{c}" for c in dom])
    p_lr = float(model["logreg"].predict_proba(row)[:, 1][0])
    p_mlp = float(model["mlp"].predict_proba(row)[:, 1][0])
    return {"logreg": p_lr, "mlp": p_mlp, "ensemble": (p_lr + p_mlp) / 2}


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
feats = load_features()
countries = sorted(feats.index)

st.sidebar.title("DefDex 🛡️")
st.sidebar.caption("Military capability comparison & analysis")

a = st.sidebar.selectbox("Country A", countries, index=countries.index("India"))
b = st.sidebar.selectbox("Country B", countries, index=countries.index("China"))
section = st.sidebar.radio(
    "Section",
    ["Overview", "Capability Radar", "Win Probability",
     "Gap & Recommendations", "Capability Clusters"],
)
if a == b:
    st.sidebar.warning("Pick two different countries.")
st.sidebar.markdown("---")
st.sidebar.caption("Data: GFP · World Bank · UCDP · SIPRI")

scores = load_scores()
clusters = load_clusters()


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #
def section_overview() -> None:
    st.title(f"{a} vs {b}")
    win = win_probability(load_win_model(), feats, a, b)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"{a} strength score", f"{feats.loc[a, 'strength_score']:.3f}")
    c2.metric(f"{b} strength score", f"{feats.loc[b, 'strength_score']:.3f}")
    rank_a = (feats["strength_score"].rank(ascending=False).astype(int))[a]
    rank_b = (feats["strength_score"].rank(ascending=False).astype(int))[b]
    c3.metric(f"{a} global rank", f"#{rank_a} / {len(feats)}")
    c4.metric(f"P({a} prevails)", f"{win['ensemble']:.0%}")

    st.markdown(
        f"**{a}** sits in **{clusters.loc[a, 'tier']}** and ranks **#{rank_a}** of "
        f"{len(feats)} by capability; **{b}** is **{clusters.loc[b, 'tier']}**, rank **#{rank_b}**."
    )
    st.info(
        "Win probability is a **capability-advantage** measure (calibrated force-balance "
        "gap), not a forecast of actual war outcomes — historical outcomes are not "
        "predictable from capability alone.",
        icon="ℹ️",
    )
    st.subheader("Capability profile")
    st.plotly_chart(radar_fig(), width="stretch")


def radar_fig() -> go.Figure:
    fig = go.Figure()
    for country, color in [(a, "#c0392b"), (b, "#2980b9")]:
        vals = radar_values(feats, country)
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]], theta=RADAR_AXES + [RADAR_AXES[0]],
            fill="toself", name=country, line_color=color))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True, height=480, margin=dict(t=30, b=30))
    return fig


def section_radar() -> None:
    st.title("Capability Radar")
    st.caption("Six-domain profile, normalized 0–1 across all 50 countries. "
               "Import dependency is shown inverted as Supply Security (higher = better).")
    st.plotly_chart(radar_fig(), width="stretch")

    tbl = pd.DataFrame({
        lbl: [feats.loc[a, f], feats.loc[b, f]]
        for f, lbl in DOMAIN_LABELS.items()
    }, index=[a, b]).T.round(3)
    st.dataframe(tbl, width="stretch")


def section_winprob() -> None:
    st.title("Win Probability")
    win = win_probability(load_win_model(), feats, a, b)
    g = go.Figure(go.Indicator(
        mode="gauge+number", value=win["ensemble"] * 100,
        number={"suffix": "%"}, title={"text": f"P({a} defeats {b})"},
        gauge={"axis": {"range": [0, 100]},
               "bar": {"color": "#c0392b"},
               "steps": [{"range": [0, 50], "color": "#f2f2f2"},
                         {"range": [50, 100], "color": "#e8f5e9"}],
               "threshold": {"line": {"color": "black", "width": 3},
                             "thickness": 0.8, "value": 50}}))
    g.update_layout(height=360, margin=dict(t=60, b=10))
    st.plotly_chart(g, width="stretch")

    c1, c2, c3 = st.columns(3)
    c1.metric("Logistic Regression", f"{win['logreg']:.0%}")
    c2.metric("Neural Net (MLP)", f"{win['mlp']:.0%}")
    c3.metric("Ensemble", f"{win['ensemble']:.0%}")
    st.info("Capability-advantage estimate from a doctrine-weighted force balance — "
            "not a forecast. Terrain, strategy, alliances and resolve are not modeled.",
            icon="ℹ️")


def section_gap() -> None:
    st.title(f"Gap Analysis — {a} vs {b}")
    rows = []
    for f, lbl in DOMAIN_LABELS.items():
        s, bb = feats.loc[a, f], feats.loc[b, f]
        disadvantage = (s - bb) if f == "import_dependency_index" else (bb - s)
        rows.append({"Domain": lbl, a: round(s, 3), b: round(bb, 3),
                     "disadvantage": round(disadvantage, 3)})
    gap = pd.DataFrame(rows).sort_values("disadvantage")
    colors = ["#27ae60" if v <= 0 else "#c0392b" for v in gap["disadvantage"]]
    fig = go.Figure(go.Bar(x=gap["disadvantage"], y=gap["Domain"], orientation="h",
                           marker_color=colors))
    fig.update_layout(height=380, margin=dict(t=20, b=30),
                      xaxis_title=f"{a}'s disadvantage vs {b}  (right = {a} behind)")
    st.plotly_chart(fig, width="stretch")

    if a == "India" and b == "China":
        st.subheader("Ranked recommendations for India")
        for _, r in load_recommendations().iterrows():
            with st.expander(f"#{int(r['rank'])} · {r['domain'].title()} "
                             f"(gap {r['disadvantage']:.3f})"):
                st.write(r["recommendation"])
                if isinstance(r["specific_targets"], str) and r["specific_targets"]:
                    st.caption("Specific targets: " + r["specific_targets"])
    else:
        st.caption("Ranked recommendations are tuned for the India vs China case "
                   "(select India / China to view them).")


def section_clusters() -> None:
    st.title("Capability Clusters")
    st.caption("KMeans tiers over the six capability domains, projected to 2D via PCA.")
    dom = list(DOMAIN_LABELS)
    X = clusters[dom].fillna(clusters[dom].median(numeric_only=True))
    xy = PCA(n_components=2, random_state=42).fit_transform(StandardScaler().fit_transform(X))
    plot = pd.DataFrame(xy, columns=["PC1", "PC2"], index=clusters.index)
    plot["tier"] = clusters["tier"]

    fig = go.Figure()
    for tier in sorted(plot["tier"].unique()):
        m = plot["tier"] == tier
        fig.add_trace(go.Scatter(
            x=plot.loc[m, "PC1"], y=plot.loc[m, "PC2"], mode="markers",
            name=tier, text=plot.index[m], marker=dict(size=9)))
    for country, color in [(a, "#c0392b"), (b, "#2980b9")]:
        fig.add_trace(go.Scatter(
            x=[plot.loc[country, "PC1"]], y=[plot.loc[country, "PC2"]],
            mode="markers+text", text=[country], textposition="top center",
            marker=dict(size=16, color=color, symbol="star"), showlegend=False))
    fig.update_layout(height=560, margin=dict(t=20, b=20),
                      xaxis_title="PC1", yaxis_title="PC2")
    st.plotly_chart(fig, width="stretch")

    tier_a = clusters.loc[a, "tier"]
    peers = [c for c in clusters.index[clusters["tier"] == tier_a] if c != a]
    st.markdown(f"**{a}** is in **{tier_a}**. Peers: {', '.join(peers[:12])}"
                + ("…" if len(peers) > 12 else ""))


SECTIONS = {
    "Overview": section_overview,
    "Capability Radar": section_radar,
    "Win Probability": section_winprob,
    "Gap & Recommendations": section_gap,
    "Capability Clusters": section_clusters,
}
SECTIONS[section]()

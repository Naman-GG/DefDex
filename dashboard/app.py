"""
DefDex — Stage 9: Streamlit Dashboard
=====================================

Interactive military-capability comparison dashboard. Leads with the India vs
China / Pakistan story but lets the user compare any two of the 50 countries.

Sections (sidebar):
    Overview            headline scores, tier, win-probability snapshot
    Capability Radar    six-domain profile, A vs B
    Service Branches    Air Force / Navy / Army platform-level comparison
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

# Service-branch breakdown: headline metric + platform-level components.
BRANCHES = {
    "Air Force": {
        "headline": ("total_aircraft", "Total aircraft"),
        "parts": {"fighter_aircraft": "Fighters", "attack_aircraft": "Attack aircraft",
                  "attack_helicopters": "Attack helicopters", "total_helicopters": "Helicopters"},
    },
    "Navy": {
        "headline": ("total_naval", "Total naval assets"),
        "parts": {"submarines": "Submarines", "aircraft_carriers": "Aircraft carriers",
                  "destroyers": "Destroyers", "frigates": "Frigates"},
    },
    "Army": {
        "headline": ("active_personnel", "Active personnel"),
        "parts": {"tanks": "Tanks", "self_propelled_artillery": "Self-propelled artillery",
                  "towed_artillery": "Towed artillery"},
    },
}

# --- Sage / olive palette ---------------------------------------------------
OLIVE = "#808000"        # buttons / primary accent
OLIVE_DRAB = "#6B8E23"   # hover / secondary accent / country B series
DARK_OLIVE = "#556B2F"   # deep green series
SAGE = "#A3B18A"         # light sage series
CRIMSON = "#C41E3A"      # country A series (high-contrast accent)
CLAY = "#BC6C25"         # "behind" / shortfall
INK = "#2F3A2A"          # chart text
TIER_COLORS = ["#556B2F", "#6B8E23", "#A3B18A", "#C2C5AA", "#8A9A5B", "#DCE2CF"]


def fmt_count(v: float) -> str:
    """Compact human-readable count (1.4M, 4,614, etc.)."""
    if pd.isna(v):
        return "n/a"
    if abs(v) >= 1e9:
        return f"{v/1e9:.1f}B"
    if abs(v) >= 1e6:
        return f"{v/1e6:.2f}M"
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    return f"{v:.0f}"

st.set_page_config(page_title="DefDex", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"], .stMarkdown { font-family:'Inter',-apple-system,system-ui,sans-serif; }

    h1, h2, h3 { color:#3A4A2F; letter-spacing:-0.4px; font-weight:700; }
    h1 { border-bottom:2px solid #9AA882; padding-bottom:12px; margin-bottom:2.6rem; }
    /* extra gap so the title rule never crowds the first metric row */
    [data-testid="stHorizontalBlock"] { margin-top:0.4rem; }

    [data-testid="stMetric"] {
        background:#FFFFFF; border:1px solid #C9D2B6; border-left:4px solid #808000;
        border-radius:14px; padding:16px 18px; box-shadow:0 1px 3px rgba(85,107,47,0.08);
    }
    section[data-testid="stSidebar"] { background:#E2E8D5; border-right:1px solid #C9D2B6; }
    section[data-testid="stSidebar"] div[role="radiogroup"] > label {
        background:#F3F5EE; border:1px solid #C9D2B6; border-radius:10px;
        padding:10px 14px; margin-bottom:8px; width:100%; transition:all .15s ease; cursor:pointer;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
        border-color:#808000; background:#EAEFE0;
    }
    .stButton>button, .stDownloadButton>button {
        background:#808000; color:#fff; border:none; border-radius:10px; font-weight:600;
        padding:8px 18px; transition:all .15s ease;
    }
    .stButton>button:hover, .stDownloadButton>button:hover {
        background:#6B8E23; transform:translateY(-1px);
    }
    [data-testid="stExpander"] { border:1px solid #C9D2B6; border-radius:12px; background:#FFFFFF; }
    [data-testid="stDataFrame"] { border-radius:12px; overflow:hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


def style_fig(fig: go.Figure, height: int) -> go.Figure:
    """Apply the shared modern, transparent, sage-toned chart styling."""
    fig.update_layout(
        height=height, margin=dict(t=40, b=30, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=INK, family="Inter, sans-serif"),
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=-0.15),
    )
    return fig


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

st.sidebar.title("DefDex")
st.sidebar.caption("Military capability comparison & analysis")

a = st.sidebar.selectbox("Country A", countries, index=countries.index("India"))
b = st.sidebar.selectbox("Country B", countries, index=countries.index("China"))
section = st.sidebar.radio(
    "Section",
    ["Overview", "Capability Radar", "Service Branches", "Win Probability",
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
        "predictable from capability alone."
    )
    st.subheader("Capability profile")
    st.plotly_chart(radar_fig(), width="stretch")


def radar_fig() -> go.Figure:
    fig = go.Figure()
    for country, color in [(a, CRIMSON), (b, DARK_OLIVE)]:
        vals = radar_values(feats, country)
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]], theta=RADAR_AXES + [RADAR_AXES[0]],
            fill="toself", name=country, line_color=color,
            fillcolor=color, opacity=0.4))
    fig.update_layout(
        polar=dict(bgcolor="rgba(255,255,255,0.5)",
                   radialaxis=dict(visible=True, range=[0, 1], gridcolor="#C9D2B6"),
                   angularaxis=dict(gridcolor="#C9D2B6")),
        showlegend=True)
    return style_fig(fig, 480)


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
               "bar": {"color": OLIVE},
               "steps": [{"range": [0, 50], "color": "#EAEFE0"},
                         {"range": [50, 100], "color": "#CBD8B4"}],
               "threshold": {"line": {"color": INK, "width": 3},
                             "thickness": 0.8, "value": 50}}))
    st.plotly_chart(style_fig(g, 360), width="stretch")

    c1, c2, c3 = st.columns(3)
    c1.metric("Logistic Regression", f"{win['logreg']:.0%}")
    c2.metric("Neural Net (MLP)", f"{win['mlp']:.0%}")
    c3.metric("Ensemble", f"{win['ensemble']:.0%}")
    st.info("Capability-advantage estimate from a doctrine-weighted force balance — "
            "not a forecast. Terrain, strategy, alliances and resolve are not modeled.")


def section_gap() -> None:
    st.title(f"Gap Analysis — {a} vs {b}")
    rows = []
    for f, lbl in DOMAIN_LABELS.items():
        s, bb = feats.loc[a, f], feats.loc[b, f]
        disadvantage = (s - bb) if f == "import_dependency_index" else (bb - s)
        rows.append({"Domain": lbl, a: round(s, 3), b: round(bb, 3),
                     "disadvantage": round(disadvantage, 3)})
    gap = pd.DataFrame(rows).sort_values("disadvantage")
    colors = [OLIVE_DRAB if v <= 0 else CLAY for v in gap["disadvantage"]]
    fig = go.Figure(go.Bar(x=gap["disadvantage"], y=gap["Domain"], orientation="h",
                           marker_color=colors))
    fig.update_xaxes(gridcolor="#C9D2B6", zerolinecolor="#9AA882")
    fig.update_layout(xaxis_title=f"{a}'s disadvantage vs {b}  (right = {a} behind)")
    st.plotly_chart(style_fig(fig, 380), width="stretch")

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


def section_branches() -> None:
    st.title("Service Branch Comparison")
    st.caption("Air Force, Navy and Army compared platform-by-platform. "
               "Branch strength is the mean of the components, normalized 0–1 across all 50 countries.")
    tabs = st.tabs(list(BRANCHES))
    for tab, (name, spec) in zip(tabs, BRANCHES.items()):
        with tab:
            parts = spec["parts"]
            head_col, head_lbl = spec["headline"]
            norm = (feats[list(parts)] - feats[list(parts)].min()) / \
                   (feats[list(parts)].max() - feats[list(parts)].min())
            idx = norm.mean(axis=1)

            c1, c2, c3 = st.columns(3)
            c1.metric(f"{a} · {head_lbl}", fmt_count(feats.loc[a, head_col]))
            c2.metric(f"{b} · {head_lbl}", fmt_count(feats.loc[b, head_col]))
            lead = a if idx[a] >= idx[b] else b
            c3.metric(f"{name} strength (0–1)", f"{idx[a]:.2f} vs {idx[b]:.2f}",
                      delta=f"{lead} leads")

            labels = list(parts.values())
            fig = go.Figure()
            for country, color in [(a, CRIMSON), (b, OLIVE_DRAB)]:
                xs = [feats.loc[country, k] for k in parts]
                fig.add_trace(go.Bar(
                    y=labels, x=xs, name=country, orientation="h",
                    marker_color=color, text=[fmt_count(x) for x in xs],
                    textposition="auto"))
            fig.update_xaxes(gridcolor="#C9D2B6", zerolinecolor="#9AA882")
            fig.update_layout(barmode="group", xaxis_title="Platform count")
            st.plotly_chart(style_fig(fig, 360), width="stretch")


def section_clusters() -> None:
    st.title("Capability Clusters")
    st.caption("KMeans tiers over the six capability domains, projected to 2D via PCA.")
    dom = list(DOMAIN_LABELS)
    X = clusters[dom].fillna(clusters[dom].median(numeric_only=True))
    xy = PCA(n_components=2, random_state=42).fit_transform(StandardScaler().fit_transform(X))
    plot = pd.DataFrame(xy, columns=["PC1", "PC2"], index=clusters.index)
    plot["tier"] = clusters["tier"]

    fig = go.Figure()
    for i, tier in enumerate(sorted(plot["tier"].unique())):
        m = plot["tier"] == tier
        fig.add_trace(go.Scatter(
            x=plot.loc[m, "PC1"], y=plot.loc[m, "PC2"], mode="markers",
            name=tier, text=plot.index[m],
            marker=dict(size=10, color=TIER_COLORS[i % len(TIER_COLORS)],
                        line=dict(width=0.5, color="#FFFFFF"))))
    for country, color in [(a, DARK_OLIVE), (b, CLAY)]:
        fig.add_trace(go.Scatter(
            x=[plot.loc[country, "PC1"]], y=[plot.loc[country, "PC2"]],
            mode="markers+text", text=[country], textposition="top center",
            marker=dict(size=18, color=color, symbol="star",
                        line=dict(width=1, color="#FFFFFF")), showlegend=False))
    fig.update_xaxes(gridcolor="#C9D2B6", zerolinecolor="#C9D2B6")
    fig.update_yaxes(gridcolor="#C9D2B6", zerolinecolor="#C9D2B6")
    fig.update_layout(xaxis_title="PC1", yaxis_title="PC2")
    st.plotly_chart(style_fig(fig, 560), width="stretch")

    tier_a = clusters.loc[a, "tier"]
    peers = [c for c in clusters.index[clusters["tier"] == tier_a] if c != a]
    st.markdown(f"**{a}** is in **{tier_a}**. Peers: {', '.join(peers[:12])}"
                + ("…" if len(peers) > 12 else ""))


SECTIONS = {
    "Overview": section_overview,
    "Capability Radar": section_radar,
    "Service Branches": section_branches,
    "Win Probability": section_winprob,
    "Gap & Recommendations": section_gap,
    "Capability Clusters": section_clusters,
}
SECTIONS[section]()

# DefDex 🛡️
### A Machine Learning Pipeline for Military Capability Comparison & War Outcome Prediction

**Live demo → [defdex.streamlit.app](https://defdex.streamlit.app/)**

> Comparing India's defense capabilities against its primary adversaries (China & Pakistan) using real open-source data, explainable ML models, and a live Streamlit dashboard.

---

## Project Status — Complete (10 of 10) ✅

| Phase | Status |
|---|---|
| Environment setup & scaffolding | ✅ Complete |
| Data collection (SIPRI, GFP, UCDP, World Bank) | ✅ Complete |
| Exploratory Data Analysis | ✅ Complete |
| Feature Engineering — 6 domain vectors | ✅ Complete |
| Training set — 50-country universe (X, y) | ✅ Complete |
| Capability scorer (XGBoost + SHAP) | ✅ Complete |
| Win-probability model (LogReg + MLP) | ✅ Complete |
| Gap Analyzer & Recommendations Engine | ✅ Complete |
| Streamlit Dashboard | ✅ Complete |
| Polish & Documentation | ✅ Complete |

---

## What is DefDex?

DefDex is a multi-model ML pipeline that quantifies and compares the military
capabilities of the **top 50 countries by Global Firepower index** across 6
domains, with India vs China & Pakistan as the analysis focus:

- **Weaponry** — platform inventories (aircraft, naval, armor, artillery) → weighted 0–1 index
- **Manpower** — active + reserve personnel, mobilization intensity
- **Geopolitics** — arms-import dependency & defense-export industry (SIPRI/World Bank TIV)
- **Terrain & Geography** — land-border and coastline exposure
- **Economic Resilience** — GDP, defense budget, energy production, spend burden
- **Historical Conflict** — interstate conflict experience encoded from UCDP data

The pipeline trains a **capability scorer** (XGBoost + SHAP), a **win-probability
model** (LogReg + MLP), and a **gap analyzer** (KMeans + recommendations), all
surfaced through an interactive **Streamlit dashboard**.

---

## Data Sources

| Dataset | Source | What it provides |
|---|---|---|
| Military Expenditure | [SIPRI](https://sipri.org/databases/milex) | Defense spend by country 2000–2023 |
| Arms Transfers | [SIPRI](https://sipri.org/databases/armstransfers) | Weapon imports/exports, TIV values |
| Global Firepower Index | [GFP](https://globalfirepower.com) | 21 military metrics per country (50-country scrape) |
| Armed Conflict Dataset | [UCDP](https://ucdp.uu.se/downloads) | 2,752 conflict events, global interstate wars |
| GDP, economics & arms trade | [World Bank API](https://data.worldbank.org) | GDP, population, mil spend %, arms imports/exports (SIPRI TIV) |

---

## Key Findings

### Modeling results

**Capability ranking (strength score, 0–1):** the XGBoost scorer (5-fold CV
R² = 0.74) ranks the US #1, then China, India (#4), with Pakistan #14 of 50.
SHAP attributes India's score primarily to weaponry and economic resilience,
with manpower and conflict experience as relative strengths.

**Win probability (capability-advantage, calibrated — not a forecast):**
- India vs China — **17%** (China dominant)
- India vs Pakistan — **68%** (India favored)
- China vs Pakistan — **92%**

**India's gap to China (Stage 8):** largest shortfalls are **weaponry**
(self-propelled artillery 100 vs 2,940; submarines 18 vs 61), **economic
resilience** (GDP $3.9T vs $18.7T), and **arms-import dependency** (India imports
~14× what China does). India *leads* China on conflict experience and manpower.

**Negative result worth noting:** training the war-outcome model on *real*
historical outcomes (Correlates of War) showed current capability differences do
**not** predict who actually won wars (CV AUC ≈ 0.5) — hence the capability-
advantage framing.

### EDA highlights

- **Defense spend (2023):** China ~$330B (steep growth since 2015), India ~$90B
  (gap with China widening), Pakistan ~$10B (flat, economy-constrained).
- **Spend as % of GDP:** Pakistan ~3.5% (highest strain) > India ~2.3% > China
  ~1.7% (lowest ratio, largest absolute budget).
- **UCDP conflict history:** India and Pakistan are the most conflict-active of
  the three, dominated by the recurring India–Pakistan dyad.

---

## Pipeline Architecture

```
 RAW DATA                      FEATURE LAYER            MODELS                    OUTPUT
 ────────                      ─────────────            ──────                    ──────
 GFP scrape ───┐                                   ┌─ XGBoost scorer ─ SHAP ─┐
 World Bank ───┼─► build_features.py ─► features ──┼─ LogReg+MLP win-prob ───┼─► Streamlit
 UCDP       ───┘    (6 domains, 50 ctry)  matrix   └─ KMeans gap analyzer ───┘   dashboard
 SIPRI      ───┘                                       + recommendations
```

Each stage writes a CSV/PKL artifact to `data/processed/` or `model/`, so stages
run independently and the dashboard reads precomputed outputs (with live
win-probability inference). See **Reproducing the pipeline** below.

---

## Project Structure

```
DefDex/
├── data/
│   ├── raw/                        # All original datasets (never modified)
│   │   ├── gfp_all_countries.csv   # 50 countries x 21 GFP metrics (Stage 5)
│   │   ├── worldbank_all_countries.csv # 50 countries x 9 WB indicators (Stage 5)
│   │   ├── ucdp_armed_conflict.csv # global conflict history
│   │   ├── sipri_milex_constant_usd.csv  # SIPRI (3-country EDA-era files)
│   │   ├── sipri_milex_gdp_share.csv
│   │   ├── sipri_tiv_india_imports.csv
│   │   ├── sipri_tiv_pak_imports.csv
│   │   ├── gfp_raw.csv             # original 3-country snapshot
│   │   └── worldbank_indicators.csv
│   └── processed/                  # ML-ready feature matrix (Stage 5)
│       ├── features.csv            # 50 countries x 45 features + target, 6 domains
│       ├── feature_dictionary.csv  # feature/domain/source/description map
│       ├── capability_scores.csv   # predicted scores + per-domain SHAP (Stage 6)
│       ├── win_probabilities.csv   # P(A beats B) for all 50x50 matchups (Stage 7)
│       ├── country_clusters.csv    # capability tiers via KMeans (Stage 8)
│       ├── india_gap_analysis.csv  # India vs China per-domain gap (Stage 8)
│       └── recommendations.csv     # ranked, data-grounded recommendations (Stage 8)
├── notebooks/
│   ├── 01_eda.ipynb                # Exploratory data analysis
│   ├── defense_spend_trend.png     # China vs India vs Pakistan absolute spend
│   └── defense_gdp_share.png      # Defense spend as % of GDP
├── src/
│   ├── data_fetcher.py             # World Bank API (original 3-country)
│   ├── fetch_worldbank.py          # World Bank API — 50-country universe
│   └── scrapers.py                 # GFP scraper prototype
├── model/
│   ├── train_capability_scorer.py  # XGBoost scorer + SHAP (Stage 6)
│   ├── capability_scorer.pkl       # trained model + metadata
│   ├── shap_summary.png            # global SHAP beeswarm
│   ├── shap_domain_contributions.png # domain drivers: India/China/Pakistan
│   ├── train_win_predictor.py      # LogReg + MLP win-probability (Stage 7)
│   ├── win_predictor.pkl           # trained ensemble + combat-power weights
│   └── win_probability_india.png   # India vs China / Pakistan win odds
├── pipeline/
│   ├── collect_gfp.py              # Scrapes GFP listing pages → 50 countries
│   ├── build_features.py           # Raw data → 6-domain feature matrix + target
│   └── gap_analyzer.py             # KMeans tiers + recommendations (Stage 8)
├── dashboard/
│   └── app.py                      # Streamlit dashboard (Stage 9)
├── requirements.txt
└── README.md
```

---

## Roadmap

```
Stage 1  ✅  Environment setup, folder scaffold, GitHub repo
Stage 2  ✅  SIPRI, GFP, arms transfer data collected
Stage 3  ✅  UCDP + World Bank data, EDA notebook, visualizations
Stage 4  ✅  Feature engineering — 6 domain vectors
Stage 5  ✅  Expanded to top-50 countries; training set (X, y) with GFP power index as target
Stage 6  ✅  XGBoost capability scorer (CV R²=0.74) + SHAP domain attributions
Stage 7  ✅  Win-probability model (LogReg + MLP) — capability-advantage, calibrated
Stage 8  ✅  Gap analyzer — KMeans tiers + ranked recommendations for India
Stage 9  ✅  Streamlit dashboard — radar, live win probability, gap, clusters
Stage 10 ✅  Polish, documentation, reproducible pipeline
```

---

## Tech Stack

```
Language:       Python 3.13
Data:           pandas, numpy
Scraping:       requests, BeautifulSoup (lxml)
Models:         scikit-learn (LogReg, MLP, KMeans), XGBoost
Explainability: SHAP
Visualization:  matplotlib, plotly
Dashboard:      Streamlit
Persistence:    joblib
```

---

## Installation

```bash
git clone https://github.com/Naman-GG/DefDex.git
cd DefDex
python3 -m venv milenv
source milenv/bin/activate
pip install -r requirements.txt

# macOS only: XGBoost needs the OpenMP runtime
brew install libomp
```

## Reproducing the pipeline

```bash
milenv/bin/python pipeline/collect_gfp.py            # scrape GFP (50 countries)
milenv/bin/python src/fetch_worldbank.py             # World Bank indicators
milenv/bin/python pipeline/build_features.py         # build feature matrix
milenv/bin/python model/train_capability_scorer.py   # train scorer + SHAP
milenv/bin/python model/train_win_predictor.py       # train win-probability model
milenv/bin/python pipeline/gap_analyzer.py           # clusters + recommendations
```

## Launch the dashboard

```bash
milenv/bin/streamlit run dashboard/app.py
```

Interactive sections: **Overview**, **Capability Radar**, **Service Branches**
(Air Force / Navy / Army), **Win Probability** (live LogReg + MLP inference for
any pair), **Gap & Recommendations**, and **Capability Clusters**. Defaults to
India vs China; any two of the 50 countries can be compared.

### Optional AI narrative (Groq)

The Gap section can render an LLM-written strategic narrative grounded in the
computed gap data. It is **off by default** and falls back to a deterministic
template narrative whenever no key is set or the API is unreachable — so the
app always works without it.

```bash
export GROQ_API_KEY=gsk_...                 # local
export GROQ_MODEL=llama-3.3-70b-versatile   # optional, overrides the default
```

On **Streamlit Community Cloud**, add `GROQ_API_KEY` under the app's *Secrets*
(`.streamlit/secrets.toml`) — because Groq is a hosted API it works in the
cloud deploy.

## Deployment

The dashboard is Streamlit Cloud-ready: relative paths, cached loaders, and no
xgboost/libomp runtime dependency (win-probability uses the sklearn model).
Point the platform at `dashboard/app.py`, install `requirements.txt`, and
optionally set `GROQ_API_KEY` as a secret to enable the AI narrative.

> **Note on the win-probability model:** It is a *capability-advantage* model — it
> quantifies the measured force-balance gap as a calibrated probability. An earlier
> experiment training on real historical outcomes (Correlates of War) found current
> capability differences do **not** predict who actually wins wars (CV AUC ≈ 0.5;
> terrain, strategy, alliances and resolve dominate), so the model answers the
> well-posed question instead of overclaiming a forecast.

---

*Built by Naman Gupta*
*Data sources: SIPRI, UCDP, Global Firepower, World Bank*

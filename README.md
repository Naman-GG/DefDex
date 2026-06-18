# DefDex 🛡️
### A Machine Learning Pipeline for Military Capability Comparison & War Outcome Prediction

> Comparing India's defense capabilities against its primary adversaries (China & Pakistan) using real open-source data, explainable ML models, and a live Streamlit dashboard.

---

## Project Status — Stage 7 of 10 ✅

| Phase | Status |
|---|---|
| Environment setup & scaffolding | ✅ Complete |
| Data collection (SIPRI, GFP, UCDP, World Bank) | ✅ Complete |
| Exploratory Data Analysis | ✅ Complete |
| Feature Engineering — 6 domain vectors | ✅ Complete |
| Training set — 50-country universe (X, y) | ✅ Complete |
| Capability scorer (XGBoost + SHAP) | ✅ Complete |
| Win-probability model (LogReg + MLP) | ✅ Complete |
| Gap Analyzer & Recommendations Engine | 🔄 Up next |
| Streamlit Dashboard | ⏳ Pending |
| Polish & Deployment | ⏳ Pending |

---

## What is DefDex?

DefDex is a multi-model ML pipeline that quantifies and compares national military capabilities across 6 domains:

- **Weaponry** — weapon system generation scoring (0–1 scale per platform)
- **Manpower** — active + reserve personnel, training quality proxy
- **Geopolitics** — alliance depth, arms import dependency, bilateral trade
- **Terrain & Geography** — border type, altitude, logistics difficulty
- **Economic Resilience** — defense % of GDP, import dependency ratio, industrial capacity
- **Historical Conflict** — past clash outcomes encoded from UCDP data

The pipeline outputs a **win-probability score** with SHAP-based explainability and a ranked list of improvement recommendations for India's defense posture.

---

## Data Sources

| Dataset | Source | What it provides |
|---|---|---|
| Military Expenditure | [SIPRI](https://sipri.org/databases/milex) | Defense spend by country 2000–2023 |
| Arms Transfers | [SIPRI](https://sipri.org/databases/armstransfers) | Weapon imports/exports, TIV values |
| Global Firepower Index | [GFP](https://globalfirepower.com) | 18 military metrics per country |
| Armed Conflict Dataset | [UCDP](https://ucdp.uu.se/downloads) | 2752 conflict events, 147 interstate wars |
| GDP & Economic Indicators | [World Bank API](https://data.worldbank.org) | GDP, population, military spend % |

---

## Key Findings So Far (EDA)

**Defense Expenditure (Constant USD, 2023):**
- China: ~$330B — near-vertical growth since 2015
- India: ~$90B — steady growth, gap with China widening
- Pakistan: ~$10B — essentially flat, constrained by economy

**Defense Spend as % of GDP:**
- Pakistan: ~3.5% — highest economic strain, military-first budget
- India: ~2.3% — declining ratio, economy growing faster than spend
- China: ~1.7% — lowest ratio but largest absolute budget

**UCDP Conflict History:**
- India: 31 conflict episodes (5 full-scale wars), 1948–2020
- Pakistan: 28 conflict episodes (3 full-scale wars), 1948–2024
- China: 19 conflict episodes (5 full-scale wars), 1949–2020

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
│       └── win_probabilities.csv   # P(A beats B) for all 50x50 matchups (Stage 7)
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
│   └── build_features.py           # Raw data → 6-domain feature matrix + target
├── dashboard/                      # Streamlit app (Stage 9+)
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
Stage 8  🔄  Gap analyzer — KMeans + recommendations engine
Stage 9       Streamlit dashboard — radar chart + win probability
Stage 10      Polish, documentation, GitHub publish
```

---

## Tech Stack

```
Language:       Python 3.13
Data:           pandas, numpy, geopandas
Models:         scikit-learn, XGBoost, PyTorch
Explainability: SHAP
Visualization:  matplotlib, plotly
Dashboard:      Streamlit
Tracking:       MLflow (Stage 6+)
Version Control: Git + DVC
```

---

## Installation

```bash
git clone https://github.com/yourusername/DefDex.git
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
```

> **Note on the win-probability model:** It is a *capability-advantage* model — it
> quantifies the measured force-balance gap as a calibrated probability. An earlier
> experiment training on real historical outcomes (Correlates of War) found current
> capability differences do **not** predict who actually wins wars (CV AUC ≈ 0.5;
> terrain, strategy, alliances and resolve dominate), so the model answers the
> well-posed question instead of overclaiming a forecast.

---

*Built by Naman Gupta*
*Data sources: SIPRI, UCDP, Global Firepower, World Bank*

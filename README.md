# DefDex 🛡️
### A Machine Learning Pipeline for Military Capability Comparison & War Outcome Prediction

> Comparing India's defense capabilities against its primary adversaries (China & Pakistan) using real open-source data, explainable ML models, and a live Streamlit dashboard.

---

## Project Status — Stage 3 of 10 ✅

| Phase | Status |
|---|---|
| Environment setup & scaffolding | ✅ Complete |
| Data collection (SIPRI, GFP, UCDP, World Bank) | ✅ Complete |
| Exploratory Data Analysis | ✅ Complete |
| Feature Engineering | 🔄 Up next |
| ML Models (XGBoost, LogReg, MLP) | ⏳ Pending |
| Gap Analyzer & Recommendations Engine | ⏳ Pending |
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
│   │   ├── sipri_milex.xlsx
│   │   ├── sipri_milex_constant_usd.csv
│   │   ├── sipri_milex_gdp_share.csv
│   │   ├── sipri_tiv_india_imports.csv
│   │   ├── sipri_tiv_pak_imports.csv
│   │   ├── india_arms_transactions_manual.csv
│   │   ├── pak_arms_transactions_manual.csv
│   │   ├── gfp_raw.csv
│   │   ├── ucdp_armed_conflict.csv
│   │   └── worldbank_indicators.csv
│   └── processed/                  # ML-ready feature vectors (Stage 4+)
├── notebooks/
│   ├── 01_eda.ipynb                # Exploratory data analysis
│   ├── defense_spend_trend.png     # China vs India vs Pakistan absolute spend
│   └── defense_gdp_share.png      # Defense spend as % of GDP
├── src/
│   ├── data_fetcher.py             # World Bank API integration
│   └── scrapers.py                 # GFP data utilities
├── model/                          # Trained models (Stage 6+)
├── pipeline/                       # Feature engineering pipeline (Stage 4+)
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
Stage 4  🔄  Feature engineering — 6 domain vectors
Stage 5       Enrich features, build training set (X, y)
Stage 6       XGBoost capability scorer + SHAP explainability
Stage 7       War outcome predictor (LogReg + MLP ensemble)
Stage 8       Gap analyzer — KMeans + recommendations engine
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
```

---

*Built by Naman Gupta*
*Data sources: SIPRI, UCDP, Global Firepower, World Bank*

"""
DefDex — Stage 5: Feature Engineering (50-country universe)
===========================================================

Turns the collected datasets into an ML-ready, per-country feature matrix
across the six capability domains, for the top-50 nations by GFP power index:

    1. Weaponry            (GFP platform counts)
    2. Manpower            (GFP personnel + labour)
    3. Geopolitics         (World Bank / SIPRI arms imports & exports)
    4. Terrain & Geography (GFP border / coastline)
    5. Economic Resilience (GFP + World Bank economics)
    6. Historical Conflict (UCDP armed-conflict history)

Target (for Stage 6 capability scorer): GFP `power_index` (lower = stronger),
plus a convenience `strength_score` (0-1, higher = stronger).

Data sources (produced by the Stage-5 collectors):
    data/raw/gfp_all_countries.csv        <- pipeline/collect_gfp.py
    data/raw/worldbank_all_countries.csv  <- src/fetch_worldbank.py
    data/raw/ucdp_armed_conflict.csv      (global, already collected)

Outputs (data/processed/):
    features.csv             one row per country, all features + target
    feature_dictionary.csv   documents every feature, domain & source

Run:
    milenv/bin/python pipeline/build_features.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

CURRENT_YEAR = 2026  # reference for recency features

_DICT: list[dict] = []


def _register(feature: str, domain: str, source: str, description: str) -> None:
    _DICT.append({"feature": feature, "domain": domain, "source": source, "description": description})


def _minmax(s: pd.Series) -> pd.Series:
    """Min-max normalize across all countries to [0, 1]; NaNs preserved."""
    lo, hi = s.min(), s.max()
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        return s.where(s.isna(), 0.5)
    return (s - lo) / (hi - lo)


# UCDP government-name token per country (substring of "Government of <token>").
# Tokens are chosen to be unambiguous within UCDP's naming (e.g. "South Korea"
# won't collide with "South Africa"). Countries with no UCDP appearance resolve
# to zero conflict involvement.
UCDP_TOKEN = {
    "United States": "United States of America", "Russia": "Russia",
    "China": "China", "India": "India", "South Korea": "South Korea",
    "France": "France", "Japan": "Japan", "United Kingdom": "United Kingdom",
    "Turkiye": "Turkey", "Italy": "Italy", "Brazil": "Brazil", "Germany": "Germany",
    "Indonesia": "Indonesia", "Pakistan": "Pakistan", "Israel": "Israel",
    "Iran": "Iran", "Australia": "Australia", "Spain": "Spain", "Egypt": "Egypt",
    "Ukraine": "Ukraine", "Poland": "Poland", "Taiwan": "Taiwan",
    "Vietnam": "Vietnam (North Vietnam)", "Thailand": "Thailand",
    "Saudi Arabia": "Saudi Arabia", "Sweden": "Sweden", "Algeria": "Algeria",
    "Canada": "Canada", "Singapore": "Singapore", "Greece": "Greece",
    "North Korea": "North Korea", "Argentina": "Argentina", "Nigeria": "Nigeria",
    "Netherlands": "Netherlands", "Myanmar": "Myanmar (Burma)", "Mexico": "Mexico",
    "Bangladesh": "Bangladesh", "Portugal": "Portugal", "Norway": "Norway",
    "South Africa": "South Africa", "Philippines": "Philippines",
    "Malaysia": "Malaysia", "Colombia": "Colombia", "Iraq": "Iraq",
    "Denmark": "Denmark", "Switzerland": "Switzerland", "Ethiopia": "Ethiopia",
    "Finland": "Finland", "Chile": "Chile", "Peru": "Peru",
}


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #
def load_sources() -> tuple[pd.DataFrame, pd.DataFrame]:
    gfp = pd.read_csv(RAW / "gfp_all_countries.csv").set_index("country")
    wb = pd.read_csv(RAW / "worldbank_all_countries.csv").set_index("country")
    return gfp, wb


# --------------------------------------------------------------------------- #
# Domain 1 — Weaponry
# --------------------------------------------------------------------------- #
def domain_weaponry(gfp: pd.DataFrame) -> pd.DataFrame:
    cols = {
        "total_aircraft": "Total aircraft fleet",
        "fighter_aircraft": "Fighter aircraft",
        "attack_aircraft": "Dedicated attack aircraft",
        "attack_helicopters": "Attack helicopters",
        "total_helicopters": "Total helicopters",
        "total_naval": "Total naval assets",
        "submarines": "Submarines",
        "aircraft_carriers": "Aircraft carriers",
        "destroyers": "Destroyers",
        "frigates": "Frigates",
        "tanks": "Main battle tanks",
        "self_propelled_artillery": "Self-propelled artillery",
        "towed_artillery": "Towed artillery",
    }
    out = pd.DataFrame(index=gfp.index)
    for c, desc in cols.items():
        out[c] = gfp[c]
        _register(c, "weaponry", "GFP", desc)

    out["combat_aircraft_ratio"] = (gfp["fighter_aircraft"] / gfp["total_aircraft"]).round(4)
    _register("combat_aircraft_ratio", "weaponry", "derived/GFP",
              "Fighters as share of total fleet (force-quality proxy)")

    # Weighted normalized composite — high-end platforms weighted up.
    weights = {
        "fighter_aircraft": 2.0, "attack_aircraft": 1.5, "attack_helicopters": 1.5,
        "submarines": 2.0, "aircraft_carriers": 2.0, "destroyers": 1.5, "frigates": 1.0,
        "tanks": 1.5, "self_propelled_artillery": 1.0, "towed_artillery": 0.5,
        "total_naval": 1.0, "total_aircraft": 1.0,
    }
    norm = pd.DataFrame({c: _minmax(out[c]) for c in weights})
    out["weaponry_index"] = (norm.mul(pd.Series(weights)).sum(axis=1) / sum(weights.values())).round(4)
    _register("weaponry_index", "weaponry", "derived",
              "Weighted min-max composite of platform inventories (high-end weighted up), 0-1")
    return out


# --------------------------------------------------------------------------- #
# Domain 2 — Manpower
# --------------------------------------------------------------------------- #
def domain_manpower(gfp: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=gfp.index)
    out["active_personnel"] = gfp["active_personnel"]
    out["reserve_personnel"] = gfp["reserve_personnel"]
    out["total_military_personnel"] = gfp["active_personnel"] + gfp["reserve_personnel"]
    out["labor_force"] = gfp["labor_force"]
    out["total_population"] = gfp["total_population"]
    _register("active_personnel", "manpower", "GFP", "Active-duty personnel")
    _register("reserve_personnel", "manpower", "GFP", "Reserve personnel")
    _register("total_military_personnel", "manpower", "derived/GFP", "Active + reserve personnel")
    _register("labor_force", "manpower", "GFP", "National labour force")
    _register("total_population", "manpower", "GFP", "Total population")

    out["reserve_ratio"] = (gfp["reserve_personnel"] / gfp["active_personnel"].replace(0, np.nan)).round(4)
    out["mil_participation_rate"] = (out["total_military_personnel"] / gfp["labor_force"]).round(6)
    _register("reserve_ratio", "manpower", "derived/GFP", "Reserve-to-active ratio (surge depth)")
    _register("mil_participation_rate", "manpower", "derived/GFP",
              "Military personnel as share of labour force (mobilization intensity)")

    out["manpower_index"] = (
        0.6 * _minmax(out["total_military_personnel"]) + 0.4 * _minmax(out["reserve_personnel"])
    ).round(4)
    _register("manpower_index", "manpower", "derived", "Composite of total + reserve personnel, 0-1")
    return out


# --------------------------------------------------------------------------- #
# Domain 3 — Geopolitics (arms trade dependency)
# --------------------------------------------------------------------------- #
def domain_geopolitics(wb: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=wb.index)
    out["arms_imports_tiv"] = wb["arms_imports_tiv"]
    out["arms_exports_tiv"] = wb["arms_exports_tiv"].fillna(0)  # no record ~ negligible exporter
    out["net_arms_trade_tiv"] = out["arms_exports_tiv"] - out["arms_imports_tiv"]
    _register("arms_imports_tiv", "geopolitics", "World Bank/SIPRI", "Arms imports (latest yr, SIPRI TIV)")
    _register("arms_exports_tiv", "geopolitics", "World Bank/SIPRI", "Arms exports (latest yr, SIPRI TIV)")
    _register("net_arms_trade_tiv", "geopolitics", "derived", "Exports - imports (positive = net exporter)")

    out["import_dependency_index"] = _minmax(out["arms_imports_tiv"]).round(4)
    out["defense_industry_index"] = _minmax(out["arms_exports_tiv"]).round(4)
    _register("import_dependency_index", "geopolitics", "derived",
              "Normalized arms-import volume (high = reliant on foreign suppliers), 0-1")
    _register("defense_industry_index", "geopolitics", "derived",
              "Normalized arms-export volume (proxy for domestic industry strength), 0-1")
    return out


# --------------------------------------------------------------------------- #
# Domain 4 — Terrain & Geography
# --------------------------------------------------------------------------- #
def domain_terrain(gfp: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=gfp.index)
    out["border_km"] = gfp["border_km"]
    out["coastline_km"] = gfp["coastline_km"]
    _register("border_km", "terrain", "GFP", "Total land border length")
    _register("coastline_km", "terrain", "GFP", "Total coastline length")

    out["coastline_ratio"] = (gfp["coastline_km"] / (gfp["coastline_km"] + gfp["border_km"]).replace(0, np.nan)).round(4)
    _register("coastline_ratio", "terrain", "derived/GFP",
              "Coastline share of total perimeter (maritime vs land-frontier exposure)")
    return out


# --------------------------------------------------------------------------- #
# Domain 5 — Economic Resilience
# --------------------------------------------------------------------------- #
def domain_economic(gfp: pd.DataFrame, wb: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=gfp.index)
    out["defense_budget_usd"] = gfp["defense_budget_usd"]
    out["oil_production_bpd"] = gfp["oil_production_bpd"]
    out["gdp_current_usd"] = wb["gdp_current_usd"]
    out["gdp_per_capita"] = wb["gdp_per_capita"].round(1)
    out["military_spend_pct_gdp"] = wb["military_spend_pct_gdp"].round(3)
    _register("defense_budget_usd", "economic", "GFP", "Annual defense budget (USD)")
    _register("oil_production_bpd", "economic", "GFP", "Domestic oil production (bbl/day, energy resilience)")
    _register("gdp_current_usd", "economic", "World Bank", "GDP, current USD (latest yr)")
    _register("gdp_per_capita", "economic", "World Bank", "GDP per capita, current USD (latest yr)")
    _register("military_spend_pct_gdp", "economic", "World Bank", "Military spend as % of GDP")

    out["defense_burden"] = (out["defense_budget_usd"] / out["gdp_current_usd"]).round(4)
    out["spend_per_soldier_usd"] = (
        gfp["defense_budget_usd"] / (gfp["active_personnel"] + gfp["reserve_personnel"]).replace(0, np.nan)
    ).round(0)
    _register("defense_burden", "economic", "derived", "Defense budget / GDP (economic strain)")
    _register("spend_per_soldier_usd", "economic", "derived", "Budget per serviceperson (capitalization proxy)")

    out["economic_resilience_index"] = (
        0.4 * _minmax(out["gdp_current_usd"])
        + 0.25 * _minmax(out["gdp_per_capita"])
        + 0.2 * _minmax(out["oil_production_bpd"])
        + 0.15 * (1 - _minmax(out["defense_burden"]))
    ).round(4)
    _register("economic_resilience_index", "economic", "derived",
              "GDP + per-capita + energy self-supply, penalized by defense burden, 0-1")
    return out


# --------------------------------------------------------------------------- #
# Domain 6 — Historical Conflict
# --------------------------------------------------------------------------- #
def domain_conflict(countries: pd.Index) -> pd.DataFrame:
    df = pd.read_csv(RAW / "ucdp_armed_conflict.csv")
    a, b = df["side_a"].fillna(""), df["side_b"].fillna("")
    out = pd.DataFrame(index=countries, dtype="float64")

    for c in countries:
        token = UCDP_TOKEN.get(c)
        if token is None:
            continue
        needle = f"Government of {token}"
        involved = df[a.str.contains(needle, regex=False) | b.str.contains(needle, regex=False)]
        interstate = involved[involved["type_of_conflict"] == 2]
        out.loc[c, "total_conflict_episodes"] = involved[["conflict_id", "year"]].drop_duplicates().shape[0]
        out.loc[c, "interstate_conflict_years"] = interstate["year"].nunique()
        out.loc[c, "high_intensity_years"] = involved[involved["intensity_level"] == 2]["year"].nunique()
        last = interstate["year"].max()
        out.loc[c, "years_since_last_interstate"] = CURRENT_YEAR - last if pd.notna(last) else np.nan

    out = out.fillna({"total_conflict_episodes": 0, "interstate_conflict_years": 0, "high_intensity_years": 0})
    _register("total_conflict_episodes", "conflict", "UCDP", "Distinct conflict-years as a party (any type)")
    _register("interstate_conflict_years", "conflict", "UCDP", "Distinct years in interstate (type 2) conflict")
    _register("high_intensity_years", "conflict", "UCDP", "Years at war-level intensity (>1000 deaths)")
    _register("years_since_last_interstate", "conflict", "derived/UCDP",
              "Years since most recent interstate conflict (NaN = none on record)")

    # Recency: recent interstate action -> high. Countries with no interstate
    # history get 0 recency (never fought a state war).
    recency = (1 - _minmax(out["years_since_last_interstate"])).fillna(0)
    out["conflict_experience_index"] = (
        0.5 * _minmax(out["interstate_conflict_years"])
        + 0.3 * _minmax(out["high_intensity_years"])
        + 0.2 * recency
    ).round(4)
    _register("conflict_experience_index", "conflict", "derived",
              "Composite of interstate exposure, high-intensity years, recency, 0-1")
    return out


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #
def build() -> pd.DataFrame:
    gfp, wb = load_sources()
    countries = gfp.index

    domains = [
        domain_weaponry(gfp),
        domain_manpower(gfp),
        domain_geopolitics(wb),
        domain_terrain(gfp),
        domain_economic(gfp, wb),
        domain_conflict(countries),
    ]
    domains = [d.reindex(countries) for d in domains]
    features = pd.concat(domains, axis=1)

    # Target: GFP power index (lower = stronger) + a 0-1 strength score.
    features.insert(0, "power_index", gfp["power_index"])
    features.insert(1, "strength_score", (1 - _minmax(gfp["power_index"])).round(4))
    _register("power_index", "target", "GFP", "GFP Power Index (lower = stronger); regression target")
    _register("strength_score", "target", "derived", "1 - normalized power index (0-1, higher = stronger)")

    features.index.name = "country"
    return features.reset_index()


def main() -> None:
    features = build()
    feat_path = PROCESSED / "features.csv"
    dict_path = PROCESSED / "feature_dictionary.csv"
    features.to_csv(feat_path, index=False)
    pd.DataFrame(_DICT).to_csv(dict_path, index=False)

    n_feat = features.shape[1] - 1
    print(f"[DefDex] Wrote {feat_path.relative_to(ROOT)}  ({features.shape[0]} countries x {n_feat} cols)")
    print(f"[DefDex] Wrote {dict_path.relative_to(ROOT)}  ({len(_DICT)} documented features)")

    miss = int(features.isna().sum().sum())
    print(f"[DefDex] Missing cells: {miss} (Taiwan economics expected NaN)")

    idx_cols = [c for c in features.columns if c.endswith("_index")]
    focus = features[features.country.isin(["India", "China", "Pakistan", "United States"])]
    print("\nDomain indices (0-1, normalized across 50 countries):")
    print(focus[["country", "strength_score", *idx_cols]].to_string(index=False))


if __name__ == "__main__":
    main()

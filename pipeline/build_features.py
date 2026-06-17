"""
DefDex — Stage 4: Feature Engineering
=====================================

Turns the raw datasets in `data/raw/` into ML-ready, per-country feature
vectors across the six capability domains:

    1. Weaponry            (GFP platform counts)
    2. Manpower            (GFP personnel + labour)
    3. Geopolitics         (SIPRI TIV arms-import dependency)
    4. Terrain & Geography (GFP border / coastline)
    5. Economic Resilience (GFP + World Bank economics)
    6. Historical Conflict (UCDP armed-conflict history)

Comparison units: India, China, Pakistan (one row each).

Outputs (written to data/processed/):
    - features.csv             one row per country, raw + normalized features
    - feature_dictionary.csv   documents every feature, its domain & source

Run:
    milenv/bin/python pipeline/build_features.py
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Paths & constants
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

COUNTRIES = ["India", "China", "Pakistan"]
CURRENT_YEAR = 2026  # reference year for "years since" recency features

# Feature dictionary accumulates (feature, domain, source, description) rows as
# domain builders register their outputs.
_DICT: list[dict] = []


def _register(feature: str, domain: str, source: str, description: str) -> None:
    _DICT.append(
        {"feature": feature, "domain": domain, "source": source, "description": description}
    )


def _minmax(s: pd.Series) -> pd.Series:
    """Min-max normalize a series across the (3) countries to [0, 1].

    NaNs are preserved. A flat series (no spread) maps to 0.5 (neutral)."""
    lo, hi = s.min(), s.max()
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        return s.where(s.isna(), 0.5)
    return (s - lo) / (hi - lo)


# --------------------------------------------------------------------------- #
# Source loaders
# --------------------------------------------------------------------------- #
def load_gfp() -> pd.DataFrame:
    """GFP is wide (metric x country); return it country-indexed (country x metric)."""
    gfp = pd.read_csv(RAW / "gfp_raw.csv").set_index("metric")
    gfp = gfp.T  # rows = india/china/pakistan, cols = metrics
    gfp.index = gfp.index.str.capitalize()
    return gfp.apply(pd.to_numeric, errors="coerce")


def _latest_sipri(path: Path) -> pd.Series:
    """Latest valid (non-'...') yearly value per country from a SIPRI wide file."""
    df = pd.read_csv(path).set_index("Country").drop(columns=["Notes"], errors="ignore")
    df = df.replace("...", np.nan).apply(pd.to_numeric, errors="coerce")
    # ffill across years then take the last column = most recent known value
    latest = df.ffill(axis=1).iloc[:, -1]
    latest.index = latest.index.str.strip()
    return latest


# --------------------------------------------------------------------------- #
# Domain 1 — Weaponry
# --------------------------------------------------------------------------- #
def domain_weaponry(gfp: pd.DataFrame) -> pd.DataFrame:
    """Hardware inventory + a weighted, normalized weaponry composite.

    Combat-capable platforms (combat aircraft, attack helis, subs, tanks,
    rocket artillery) are weighted above raw totals to reward quality of force."""
    raw_cols = {
        "total_aircraft": "Total aircraft fleet",
        "combat_aircraft": "Combat-capable aircraft",
        "attack_helicopters": "Attack helicopters",
        "total_naval": "Total naval assets",
        "submarines": "Submarines",
        "total_land_vehicles": "Armoured land vehicles",
        "tanks": "Main battle tanks",
        "artillery": "Towed/self-propelled artillery",
        "rocket_artillery": "Rocket artillery (MLRS)",
    }
    out = pd.DataFrame(index=gfp.index)
    for col, desc in raw_cols.items():
        out[col] = gfp[col]
        _register(col, "weaponry", "GFP", desc)

    out["combat_aircraft_ratio"] = (gfp["combat_aircraft"] / gfp["total_aircraft"]).round(4)
    _register("combat_aircraft_ratio", "weaponry", "derived/GFP",
              "Combat-capable aircraft as share of total fleet (force-quality proxy)")

    # Weighted normalized composite — combat platforms count double.
    weights = {
        "combat_aircraft": 2.0, "attack_helicopters": 1.5, "submarines": 2.0,
        "tanks": 1.5, "rocket_artillery": 1.5, "total_naval": 1.0,
        "total_aircraft": 1.0, "artillery": 1.0, "total_land_vehicles": 1.0,
    }
    norm = pd.DataFrame({c: _minmax(out[c]) for c in weights})
    out["weaponry_index"] = (
        norm.mul(pd.Series(weights)).sum(axis=1) / sum(weights.values())
    ).round(4)
    _register("weaponry_index", "weaponry", "derived",
              "Weighted min-max composite of platform inventories (combat platforms weighted higher), 0-1")
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
    _register("active_personnel", "manpower", "GFP", "Active-duty military personnel")
    _register("reserve_personnel", "manpower", "GFP", "Reserve military personnel")
    _register("total_military_personnel", "manpower", "derived/GFP", "Active + reserve personnel")
    _register("labor_force", "manpower", "GFP", "National labour force")
    _register("total_population", "manpower", "GFP", "Total population")

    out["reserve_ratio"] = (gfp["reserve_personnel"] / gfp["active_personnel"]).round(4)
    out["mil_participation_rate"] = (out["total_military_personnel"] / gfp["labor_force"]).round(6)
    _register("reserve_ratio", "manpower", "derived/GFP", "Reserve-to-active personnel ratio (surge depth)")
    _register("mil_participation_rate", "manpower", "derived/GFP",
              "Total military personnel as share of labour force (mobilization intensity)")

    out["manpower_index"] = (
        0.6 * _minmax(out["total_military_personnel"]) + 0.4 * _minmax(out["reserve_personnel"])
    ).round(4)
    _register("manpower_index", "manpower", "derived",
              "Composite of total + reserve personnel (normalized), 0-1")
    return out


# --------------------------------------------------------------------------- #
# Domain 3 — Geopolitics (arms-import dependency)
# --------------------------------------------------------------------------- #
def _parse_tiv_imports(path: Path) -> dict:
    """Parse a SIPRI TIV import file -> supplier concentration metrics.

    The file has a multi-line header preamble; the real table starts at the
    'Supplier' row. We use the 'Sum total years' column for each supplier and
    the 'Total exports to <country>' row as the denominator."""
    raw = pd.read_csv(path, skiprows=8)
    raw.columns = [str(c).strip() for c in raw.columns]
    raw = raw[raw["Supplier"].notna()].copy()

    total_row = raw[raw["Supplier"].str.contains("Total", case=False, na=False)]
    suppliers = raw[~raw["Supplier"].str.contains("Total", case=False, na=False)].copy()

    suppliers["sum"] = pd.to_numeric(suppliers["Sum total years"], errors="coerce").fillna(0)
    suppliers = suppliers[suppliers["sum"] > 0]

    total = pd.to_numeric(total_row["Sum total years"], errors="coerce").iloc[0] \
        if not total_row.empty else suppliers["sum"].sum()
    shares = suppliers["sum"] / total
    return {
        "arms_supplier_count": int((suppliers["sum"] > 0).sum()),
        "top_supplier_share": round(float(shares.max()), 4),
        "supplier_hhi": round(float((shares ** 2).sum()), 4),  # 0..1, higher = concentrated
        "total_tiv_imports": round(float(total), 1),
    }


def domain_geopolitics(gfp: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=gfp.index, dtype="float64")
    files = {
        "India": RAW / "sipri_tiv_india_imports.csv",
        "Pakistan": RAW / "sipri_tiv_pak_imports.csv",
    }
    for country, path in files.items():
        for k, v in _parse_tiv_imports(path).items():
            out.loc[country, k] = v

    # China: no TIV import file collected. China is a net arms EXPORTER with a
    # largely domestic supply base, so import-dependency metrics are not
    # applicable rather than zero. Left as NaN and flagged in data_note.
    out["data_note"] = ""
    out.loc["China", "data_note"] = "TIV imports not collected (net exporter, domestic supply)"

    _register("arms_supplier_count", "geopolitics", "SIPRI TIV", "Number of distinct arms suppliers")
    _register("top_supplier_share", "geopolitics", "derived/SIPRI",
              "Largest single supplier's share of imports (single-source risk)")
    _register("supplier_hhi", "geopolitics", "derived/SIPRI",
              "Herfindahl index of supplier concentration (0-1; higher = more concentrated)")
    _register("total_tiv_imports", "geopolitics", "SIPRI TIV",
              "Cumulative arms imports in SIPRI trend-indicator values")

    # Import-dependency index: high import volume + high concentration = fragile.
    # Normalized only over countries with data (India, Pakistan).
    dep = 0.5 * _minmax(out["total_tiv_imports"]) + 0.5 * out["supplier_hhi"]
    out["import_dependency_index"] = dep.round(4)
    out["self_reliance_index"] = (1 - dep).round(4)
    _register("import_dependency_index", "geopolitics", "derived",
              "Arms-import fragility: blend of import volume + supplier concentration, 0-1 (China NaN)")
    _register("self_reliance_index", "geopolitics", "derived",
              "1 - import_dependency_index (domestic supply security, China NaN)")
    _register("data_note", "geopolitics", "meta", "Data-quality caveat per country")
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

    out["coastline_ratio"] = (gfp["coastline_km"] / (gfp["coastline_km"] + gfp["border_km"])).round(4)
    _register("coastline_ratio", "terrain", "derived/GFP",
              "Coastline share of total perimeter (maritime vs land-frontier exposure)")
    return out


# --------------------------------------------------------------------------- #
# Domain 5 — Economic Resilience
# --------------------------------------------------------------------------- #
def domain_economic(gfp: pd.DataFrame) -> pd.DataFrame:
    wb = pd.read_csv(RAW / "worldbank_indicators.csv")
    latest = wb.sort_values("year").groupby("country").last()  # most recent year per country

    out = pd.DataFrame(index=gfp.index)
    out["defense_budget_usd"] = gfp["defense_budget_usd"]
    out["oil_production_bpd"] = gfp["oil_production_bpd"]
    out["gdp_current_usd"] = latest["gdp_current_usd"]
    out["gdp_per_capita"] = latest["gdp_per_capita"].round(1)
    # Prefer SIPRI GDP-share (defense-specific) when available; fall back to WB.
    sipri_share = _latest_sipri(RAW / "sipri_milex_gdp_share.csv") * 100  # fraction -> %
    out["military_spend_pct_gdp"] = sipri_share.reindex(out.index).round(3)
    out["military_spend_pct_gdp"] = out["military_spend_pct_gdp"].fillna(
        latest["military_spend_pct_gdp"]
    )
    _register("defense_budget_usd", "economic", "GFP", "Annual defense budget (USD)")
    _register("oil_production_bpd", "economic", "GFP", "Domestic oil production (barrels/day, energy resilience)")
    _register("gdp_current_usd", "economic", "World Bank", "GDP, current USD (latest year)")
    _register("gdp_per_capita", "economic", "World Bank", "GDP per capita, current USD (latest year)")
    _register("military_spend_pct_gdp", "economic", "SIPRI/World Bank", "Military spend as % of GDP")

    out["defense_burden"] = (out["defense_budget_usd"] / out["gdp_current_usd"]).round(4)
    out["spend_per_soldier_usd"] = (
        gfp["defense_budget_usd"] / (gfp["active_personnel"] + gfp["reserve_personnel"])
    ).round(0)
    _register("defense_burden", "economic", "derived",
              "Defense budget / GDP (economic strain of current spend)")
    _register("spend_per_soldier_usd", "economic", "derived",
              "Defense budget per total serviceperson (capitalization proxy)")

    # Resilience: large economy + rich per-capita + energy self-supply, penalized
    # by a heavy defense burden (less headroom to surge).
    out["economic_resilience_index"] = (
        0.4 * _minmax(out["gdp_current_usd"])
        + 0.25 * _minmax(out["gdp_per_capita"])
        + 0.2 * _minmax(out["oil_production_bpd"])
        + 0.15 * (1 - _minmax(out["defense_burden"]))
    ).round(4)
    _register("economic_resilience_index", "economic", "derived",
              "Composite: GDP + per-capita + energy self-supply, penalized by defense burden, 0-1")
    return out


# --------------------------------------------------------------------------- #
# Domain 6 — Historical Conflict
# --------------------------------------------------------------------------- #
def domain_conflict() -> pd.DataFrame:
    df = pd.read_csv(RAW / "ucdp_armed_conflict.csv")
    out = pd.DataFrame(index=pd.Index(COUNTRIES, name="country"), dtype="float64")

    for c in COUNTRIES:
        involved = df[
            df["side_a"].str.contains(c, case=False, na=False)
            | df["side_b"].str.contains(c, case=False, na=False)
        ]
        interstate = involved[involved["type_of_conflict"] == 2]
        out.loc[c, "total_conflict_episodes"] = involved[["conflict_id", "year"]].drop_duplicates().shape[0]
        out.loc[c, "interstate_conflict_years"] = interstate["year"].nunique()
        out.loc[c, "high_intensity_years"] = involved[involved["intensity_level"] == 2]["year"].nunique()
        last = interstate["year"].max()
        out.loc[c, "years_since_last_interstate"] = (
            CURRENT_YEAR - last if pd.notna(last) else np.nan
        )

    _register("total_conflict_episodes", "conflict", "UCDP", "Distinct conflict-years country was a party to (any type)")
    _register("interstate_conflict_years", "conflict", "UCDP", "Distinct years in state-vs-state (type 2) conflict")
    _register("high_intensity_years", "conflict", "UCDP", "Years at war-level intensity (>1000 battle deaths)")
    _register("years_since_last_interstate", "conflict", "derived/UCDP", "Years since most recent interstate conflict (recency)")

    # Experience index: more interstate + high-intensity exposure and more
    # recent action => higher battle-tested score.
    recency = 1 - _minmax(out["years_since_last_interstate"])  # recent -> high
    out["conflict_experience_index"] = (
        0.5 * _minmax(out["interstate_conflict_years"])
        + 0.3 * _minmax(out["high_intensity_years"])
        + 0.2 * recency
    ).round(4)
    _register("conflict_experience_index", "conflict", "derived",
              "Composite of interstate exposure, high-intensity years, and recency, 0-1")
    return out


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #
def build() -> pd.DataFrame:
    gfp = load_gfp()
    domains = [
        domain_weaponry(gfp),
        domain_manpower(gfp),
        domain_geopolitics(gfp),
        domain_terrain(gfp),
        domain_economic(gfp),
        domain_conflict(),
    ]
    # Align everyone on the canonical country order.
    domains = [d.reindex(COUNTRIES) for d in domains]
    features = pd.concat(domains, axis=1)
    features.index.name = "country"
    features = features.reset_index()
    return features


def main() -> None:
    features = build()
    feat_path = PROCESSED / "features.csv"
    dict_path = PROCESSED / "feature_dictionary.csv"
    features.to_csv(feat_path, index=False)
    pd.DataFrame(_DICT).to_csv(dict_path, index=False)

    n_feat = features.shape[1] - 1  # minus the country column
    print(f"[DefDex] Wrote {feat_path.relative_to(ROOT)}  ({features.shape[0]} countries x {n_feat} features)")
    print(f"[DefDex] Wrote {dict_path.relative_to(ROOT)}  ({len(_DICT)} documented features)")

    idx_cols = [c for c in features.columns if c.endswith("_index")]
    print("\nDomain indices (0-1, normalized across the 3 countries):")
    print(features[["country", *idx_cols]].to_string(index=False))


if __name__ == "__main__":
    main()

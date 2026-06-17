"""
DefDex — Stage 5 data collection: World Bank indicators (50-country universe)
=============================================================================

Pulls economic, demographic and SIPRI-sourced military indicators from the
World Bank API for the same top-50 universe scraped from GFP. Crucially this
includes arms imports/exports (SIPRI TIV) for ALL countries — which fills the
China arms-import gap that the standalone SIPRI TIV files couldn't.

One batched request per indicator (all 50 ISO-3 codes joined by ';'), then the
most-recent non-null value per country. Output is a latest-snapshot wide table
to join 1:1 with gfp_all_countries.csv on country name.

Note: Taiwan (TWN) and North Korea (PRK) have little/no World Bank coverage and
will come back as NaN — expected, handled downstream.

Output: data/raw/worldbank_all_countries.csv

Run:
    milenv/bin/python src/fetch_worldbank.py
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

# GFP country name -> World Bank ISO-3 code (covers the top-50 universe)
NAME_TO_ISO3 = {
    "United States": "USA", "Russia": "RUS", "China": "CHN", "India": "IND",
    "South Korea": "KOR", "France": "FRA", "Japan": "JPN", "United Kingdom": "GBR",
    "Turkiye": "TUR", "Italy": "ITA", "Brazil": "BRA", "Germany": "DEU",
    "Indonesia": "IDN", "Pakistan": "PAK", "Israel": "ISR", "Iran": "IRN",
    "Australia": "AUS", "Spain": "ESP", "Egypt": "EGY", "Ukraine": "UKR",
    "Poland": "POL", "Taiwan": "TWN", "Vietnam": "VNM", "Thailand": "THA",
    "Saudi Arabia": "SAU", "Sweden": "SWE", "Algeria": "DZA", "Canada": "CAN",
    "Singapore": "SGP", "Greece": "GRC", "North Korea": "PRK", "Argentina": "ARG",
    "Nigeria": "NGA", "Netherlands": "NLD", "Myanmar": "MMR", "Mexico": "MEX",
    "Bangladesh": "BGD", "Portugal": "PRT", "Norway": "NOR", "South Africa": "ZAF",
    "Philippines": "PHL", "Malaysia": "MYS", "Colombia": "COL", "Iraq": "IRQ",
    "Denmark": "DNK", "Switzerland": "CHE", "Ethiopia": "ETH", "Finland": "FIN",
    "Chile": "CHL", "Peru": "PER",
}
ISO3_TO_NAME = {v: k for k, v in NAME_TO_ISO3.items()}

INDICATORS = {
    "NY.GDP.MKTP.CD":    "gdp_current_usd",
    "NY.GDP.PCAP.CD":    "gdp_per_capita",
    "MS.MIL.XPND.GD.ZS": "military_spend_pct_gdp",
    "SP.POP.TOTL":       "population_wb",
    "SL.TLF.TOTL.IN":    "labor_force_wb",
    "MS.MIL.TOTL.P1":    "armed_forces_personnel",
    "MS.MIL.XPND.CD":    "mil_exp_usd_wb",
    "MS.MIL.MPRT.KD":    "arms_imports_tiv",
    "MS.MIL.XPRT.KD":    "arms_exports_tiv",
}

BASE = "https://api.worldbank.org/v2"


def fetch_indicator(name: str) -> pd.Series:
    """Most-recent non-null value per country for one indicator, indexed by name."""
    codes = ";".join(NAME_TO_ISO3.values())
    code = {v: k for k, v in INDICATORS.items()}[name]
    url = f"{BASE}/country/{codes}/indicator/{code}"
    params = {"format": "json", "mrv": 8, "per_page": 5000}

    for attempt in range(3):
        try:
            r = requests.get(url, params=params, timeout=45)
            payload = r.json()
            rows = payload[1] if len(payload) > 1 and payload[1] else []
            break
        except Exception as e:  # noqa: BLE001
            if attempt == 2:
                print(f"[WB] {name}: FAILED after retries ({e})")
                return pd.Series(dtype="float64")
            time.sleep(2)

    # Batched responses are NOT reliably newest-first, so explicitly keep the
    # value for the highest year (latest) per country.
    best: dict[str, tuple[int, float]] = {}  # iso -> (year, value)
    for d in rows:
        iso = d.get("countryiso3code")
        val = d.get("value")
        if iso in ISO3_TO_NAME and val is not None:
            year = int(d["date"])
            if iso not in best or year > best[iso][0]:
                best[iso] = (year, float(val))
    latest = {ISO3_TO_NAME[iso]: v for iso, (_, v) in best.items()}
    return pd.Series(latest, name=name)


def main() -> None:
    cols = {}
    for name in INDICATORS.values():
        s = fetch_indicator(name)
        cols[name] = s
        print(f"[WB] {name:24s} ({s.notna().sum()}/{len(NAME_TO_ISO3)} countries)")
        time.sleep(0.5)

    df = pd.DataFrame(cols)
    df = df.reindex(NAME_TO_ISO3.keys())  # canonical 50-country order
    df.index.name = "country"

    out_path = RAW / "worldbank_all_countries.csv"
    df.reset_index().to_csv(out_path, index=False)
    print(f"\n[WB] Wrote {out_path.relative_to(ROOT)} "
          f"({df.shape[0]} countries x {df.shape[1]} indicators)")
    missing = df.index[df.isna().all(axis=1)].tolist()
    if missing:
        print(f"[WB] No WB coverage (expected): {missing}")


if __name__ == "__main__":
    main()

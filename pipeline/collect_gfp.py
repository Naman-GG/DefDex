"""
DefDex — Stage 5 data collection: Global Firepower (GFP) scraper
================================================================

GFP's per-country detail pages only expose *rankings*, not raw counts. The
raw values live on per-metric "listing" pages that table every country at
once (e.g. /armor-tanks-total.php). So we scrape ~20 listing pages (one HTTP
request per metric) rather than 50 country pages — far fewer requests and a
single, uniform parse.

Country universe: the top-50 nations by GFP Power Index (includes India,
China, Pakistan at ranks 3/4/14). Everything is keyed on country NAME because
GFP uses non-standard codes (SKO, UKD, TKY...) that don't match ISO-3.

Output: data/raw/gfp_all_countries.csv  (one row per country, wide metrics)

Run:
    milenv/bin/python pipeline/collect_gfp.py
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

BASE = "https://www.globalfirepower.com"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
}
N_COUNTRIES = 50
SLEEP = 1.5  # polite delay between requests

# feature_name -> (listing page slug, value kind)
#   kind "num"   plain integer/float value
#   kind "money" value prefixed by '$'
METRIC_PAGES: dict[str, tuple[str, str]] = {
    "active_personnel":         ("active-military-manpower.php", "num"),
    "reserve_personnel":        ("active-reserve-military-manpower.php", "num"),
    "total_aircraft":           ("aircraft-total.php", "num"),
    "fighter_aircraft":         ("aircraft-total-fighters.php", "num"),
    "attack_aircraft":          ("aircraft-total-attack-types.php", "num"),
    "attack_helicopters":       ("aircraft-helicopters-attack.php", "num"),
    "total_helicopters":        ("aircraft-helicopters-total.php", "num"),
    "total_naval":              ("navy-ships.php", "num"),
    "submarines":               ("navy-submarines.php", "num"),
    "aircraft_carriers":        ("navy-aircraft-carriers.php", "num"),
    "destroyers":               ("navy-destroyers.php", "num"),
    "frigates":                 ("navy-frigates.php", "num"),
    "tanks":                    ("armor-tanks-total.php", "num"),
    "self_propelled_artillery": ("armor-self-propelled-guns-total.php", "num"),
    "towed_artillery":          ("armor-towed-artillery-total.php", "num"),
    "defense_budget_usd":       ("defense-spending-budget.php", "money"),
    "oil_production_bpd":       ("oil-production-by-country.php", "num"),
    "labor_force":              ("labor-force-by-country.php", "num"),
    "total_population":         ("total-population-by-country.php", "num"),
    "coastline_km":             ("coastline-coverage.php", "num"),
    "border_km":                ("border-coverage.php", "num"),
}

# name row: "<Name> <GFPCODE> <value>"  (value may carry $ and commas)
_ROW = re.compile(r"([A-Z][A-Za-z .'()-]+?)\s+([A-Z]{3})\s+\$?\s*([\d][\d,]*(?:\.\d+)?)")
# power-index row: "<Name> <GFPCODE> PwrIndx: 0.xxxx"
_PWR = re.compile(r"([A-Z][A-Za-z .'()-]+?)\s+([A-Z]{3})\s+PwrIndx:\s+([\d.]+)")


def _get_text(slug: str) -> str:
    r = requests.get(f"{BASE}/{slug}", headers=HEADERS, timeout=25)
    r.raise_for_status()
    txt = BeautifulSoup(r.text, "lxml").get_text(" ", strip=True)
    return re.sub(r"\s+", " ", txt)  # collapse nbsp/tabs to single spaces


def _to_float(s: str) -> float:
    return float(s.replace(",", ""))


def get_universe() -> pd.DataFrame:
    """Top-N countries by GFP Power Index -> DataFrame(name, gfp_code, power_index)."""
    txt = _get_text("countries-listing.php")
    rows = _PWR.findall(txt)
    df = pd.DataFrame(
        [{"country": n.strip(), "gfp_code": c, "power_index": float(p)} for n, c, p in rows]
    )
    df = df.sort_values("power_index").head(N_COUNTRIES).reset_index(drop=True)
    return df


def scrape_metric(slug: str, names: set[str]) -> dict[str, float]:
    """Return {country_name: value} for the given listing page, limited to `names`."""
    txt = _get_text(slug)
    out: dict[str, float] = {}
    for name, _code, val in _ROW.findall(txt):
        name = name.strip()
        if name in names and name not in out:  # first occurrence = the ranked row
            out[name] = _to_float(val)
    return out


def main() -> None:
    universe = get_universe()
    names = set(universe["country"])
    print(f"[GFP] Universe: top {len(names)} by power index "
          f"(India/China/Pakistan included: "
          f"{all(c in names for c in ['India', 'China', 'Pakistan'])})")

    data = universe.set_index("country").copy()
    for feature, (slug, _kind) in METRIC_PAGES.items():
        try:
            vals = scrape_metric(slug, names)
            data[feature] = pd.Series(vals)
            hit = data[feature].notna().sum()
            print(f"[GFP] {feature:26s} <- {slug:42s} ({hit}/{len(names)} countries)")
        except Exception as e:  # noqa: BLE001
            print(f"[GFP] {feature:26s} FAILED on {slug}: {e}")
        time.sleep(SLEEP)

    out_path = RAW / "gfp_all_countries.csv"
    data.reset_index().to_csv(out_path, index=False)
    print(f"\n[GFP] Wrote {out_path.relative_to(ROOT)} "
          f"({data.shape[0]} countries x {data.shape[1]} columns)")


if __name__ == "__main__":
    main()

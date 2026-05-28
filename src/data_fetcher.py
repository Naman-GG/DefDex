import requests
import pandas as pd

COUNTRIES = {
    "IND": "India",
    "CHN": "China",
    "PAK": "Pakistan"
}

INDICATORS = {
    "NY.GDP.MKTP.CD": "gdp_current_usd",
    "MS.MIL.XPND.GD.ZS": "military_spend_pct_gdp",
    "SP.POP.TOTL": "total_population",
    "NY.GDP.PCAP.CD": "gdp_per_capita"
}

def fetch_indicator(country_code, indicator_code, indicator_name):
    url = f"https://api.worldbank.org/v2/country/{country_code}/indicator/{indicator_code}"
    params = {"format": "json", "mrv": 15, "per_page": 20}
    
    try:
        r = requests.get(url, params=params, timeout=10)
        data = r.json()[1]
        records = []
        for d in data:
            if d["value"] is not None:
                records.append({
                    "country": COUNTRIES[country_code],
                    "year": int(d["date"]),
                    "indicator": indicator_name,
                    "value": d["value"]
                })
        return records
    except Exception as e:
        print(f"Error fetching {indicator_name} for {country_code}: {e}")
        return []

def fetch_all():
    all_records = []
    
    for code, name in COUNTRIES.items():
        for indicator_code, indicator_name in INDICATORS.items():
            print(f"Fetching {indicator_name} for {name}...")
            records = fetch_indicator(code, indicator_code, indicator_name)
            all_records.extend(records)
    
    df = pd.DataFrame(all_records)
    
    # Pivot to wide format — one row per country per year
    df_wide = df.pivot_table(
        index=["country", "year"],
        columns="indicator",
        values="value"
    ).reset_index()
    
    df_wide.to_csv("data/raw/worldbank_indicators.csv", index=False)
    print(f"\n✅ Saved worldbank_indicators.csv — shape: {df_wide.shape}")
    print(df_wide.tail(10))
    return df_wide

if __name__ == "__main__":
    fetch_all()
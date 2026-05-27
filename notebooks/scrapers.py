import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

COUNTRIES = ["india", "china", "pakistan"]

def inspect_gfp(country_slug):
    url = f"https://www.globalfirepower.com/country-military-strength-detail.php?country_id={country_slug}"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    r = requests.get(url, headers=headers, timeout=10)
    soup = BeautifulSoup(r.text, "lxml")
    
    # Print a chunk of HTML to find actual class names
    print(f"\n{'='*50}")
    print(f"HTML SAMPLE FOR: {country_slug}")
    print(f"{'='*50}")
    print(r.text[3000:6000])
    return soup

def scrape_gfp(country_slug, soup):
    data = {"country": country_slug}
    
    # Attempt 1 — original class names
    for span in soup.select("span.countryStat"):
        label = span.find_previous("span", class_="countryStatLabel")
        if label:
            key = label.text.strip().lower().replace(" ", "_")
            val = span.text.strip().replace(",", "")
            data[key] = val

    # Attempt 2 — div based structure
    if len(data) <= 1:
        for div in soup.select("div.picData"):
            label = div.find("span", class_="textWhite")
            value = div.find("span", class_="textRed") or div.find("span", class_="textGreen")
            if label and value:
                key = label.text.strip().lower().replace(" ", "_")
                val = value.text.strip().replace(",", "")
                data[key] = val

    # Attempt 3 — table based structure
    if len(data) <= 1:
        for row in soup.select("tr"):
            cols = row.find_all("td")
            if len(cols) == 2:
                key = cols[0].text.strip().lower().replace(" ", "_")
                val = cols[1].text.strip().replace(",", "")
                if key and val:
                    data[key] = val

    return data

def scrape_all():
    results = []
    
    for country in COUNTRIES:
        print(f"\nScraping {country}...")
        
        try:
            soup = inspect_gfp(country)
            data = scrape_gfp(country, soup)
            print(f"Fields captured for {country}: {len(data)}")
            print(f"Sample keys: {list(data.keys())[:5]}")
            results.append(data)
            time.sleep(3)
            
        except Exception as e:
            print(f"Error on {country}: {e}")
            results.append({"country": country})
    
    df = pd.DataFrame(results).set_index("country")
    
    if df.shape[1] > 1:
        df.to_csv("data/raw/gfp_raw.csv")
        print(f"\n✅ Saved gfp_raw.csv — shape: {df.shape}")
    else:
        print("\n❌ Scraper still not finding data.")
        print("Paste the HTML printed above and we'll fix the selectors.")
    
    return df

if __name__ == "__main__":
    df = scrape_all()
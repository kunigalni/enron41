import requests
import pandas as pd
from dateutil import parser
from preprocessing import preprocess_emails

def fetch_all_emails(host, index, batch_size=5000):
    url = f"{host}{index}/_search?scroll=10m"

    query = {
        "size": batch_size,
        "query": {"match_all": {}}
    }

    r = requests.get(url, json=query, headers={"Content-Type": "application/json"})
    data = r.json()

    scroll_id = data["_scroll_id"]
    hits = data["hits"]["hits"]

    all_hits = hits.copy()

    while True:
        scroll_url = f"{host}_search/scroll"
        payload = {
            "scroll": "10m",
            "scroll_id": scroll_id
        }

        r = requests.get(scroll_url, json=payload, headers={"Content-Type": "application/json"})
        data = r.json()

        hits = data["hits"]["hits"]
        if not hits:
            break

        all_hits.extend(hits)
        scroll_id = data["_scroll_id"]

    # Deduplicate by _id
    unique = {h["_id"]: h for h in all_hits}
    print(f"Unique documents: {len(unique)}")

    return list(unique.values())

class Main:

    @staticmethod
    def main():
        #load dataset 
        host = "http://18.188.56.207:9200/"
        index = "enron"
        
        print("Checking index...")
        print(requests.get(host + f"_cat/indices/{index}").content)

        print("Fetching ALL emails - will take a while")
        hits = fetch_all_emails(host, index)

        # Convert to DataFrame
        df = pd.DataFrame([h["_source"] for h in hits])

        # Parse dates
        df["date"] = df["date"].apply(parser.parse)
        df["date"] = df["date"].dt.tz_localize(None)

        # Filter to Aug–Dec 2001
        start = parser.parse("2001-08-01")
        end = parser.parse("2001-12-31")
        df = df[(df["date"] >= start) & (df["date"] <= end)]
        df = df.reset_index(drop=True)

        print(f"Emails in fraud window (Aug–Dec 2001): {len(df)}")

        print("Preprocessing")
        cleaned_df = preprocess_emails(df)

        # print(cleaned_df[["subject", "clean_text", "tokens"]].head())

if __name__ == "__main__":
    Main.main()

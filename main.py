import requests
import pandas as pd
from dateutil import parser
from preprocessing import preprocess_emails
from keywordFrequency import KeywordFrequency


def fetch_all_emails(host, index, batch_size=5000):
    print(f"\n[fetch_all_emails] Starting scroll fetch for index '{index}'")
    url = f"{host}{index}/_search?scroll=10m"

    query = {
        "size": batch_size,
        "query": {"match_all": {}}
    }

    # Initial request
    r = requests.get(url, json=query, headers={"Content-Type": "application/json"})
    data = r.json()

    scroll_id = data["_scroll_id"]
    hits = data["hits"]["hits"]
    all_hits = hits.copy()

    print(f"[fetch_all_emails] Initial batch: {len(hits)} hits")

    # Scroll loop
    while True:
        scroll_url = f"{host}_search/scroll"
        payload = {"scroll": "10m", "scroll_id": scroll_id}

        r = requests.get(scroll_url, json=payload, headers={"Content-Type": "application/json"})
        data = r.json()

        hits = data["hits"]["hits"]
        if not hits:
            print("[fetch_all_emails] Scroll returned 0 hits — stopping.")
            break

        all_hits.extend(hits)
        scroll_id = data["_scroll_id"]

        if len(all_hits) % 10000 == 0:
            print(f"[fetch_all_emails] Raw hits so far: {len(all_hits)}")

    print(f"[fetch_all_emails] Total raw hits before dedupe: {len(all_hits)}")

    # Deduplicate by ES _id
    unique = {h["_id"]: h for h in all_hits}
    print(f"[fetch_all_emails] Unique documents after dedupe: {len(unique)}")

    return list(unique.values())


class Main:

    @staticmethod
    def main():
        host = "http://18.188.56.207:9200/"
        index = "enron"

        print("\n========== Checking Index ==========")
        print(requests.get(host + f"_cat/indices/{index}").content)

        # True count from ES
        count_resp = requests.get(host + f"{index}/_count").json()
        print(f"[main] /_count reports: {count_resp['count']} documents")

        print("\n========== Fetching Emails ==========")
        hits = fetch_all_emails(host, index)

        print("\n========== Converting to DataFrame ==========")
        df = pd.DataFrame([h["_source"] for h in hits])
        df = df.head(5000) #ONLY FOR TESTING SPEED PURPOSES
        print(f"[main] DataFrame created with {len(df)} rows")

        print("\n========== Parsing Dates ==========")
        df["date"] = df["date"].apply(parser.parse)
        df["date"] = df["date"].dt.tz_localize(None)

        print("\n========== Filtering Fraud Window (Aug–Dec 2001) ==========")
        start = parser.parse("2001-08-01")
        end = parser.parse("2001-12-31")

        df = df[(df["date"] >= start) & (df["date"] <= end)]
        df = df.reset_index(drop=True)

        print(f"[main] Emails in fraud window: {len(df)}")

        print("\n========== Preprocessing Emails ==========")

        # Make a deep copy so preprocess_emails cannot mutate df
        df_filtered = df.copy(deep=True)

        cleaned_df = preprocess_emails(df_filtered)

        print(f"[main] Preprocessing complete. Example:")
        print(cleaned_df[["subject", "clean_text", "tokens"]].head())
        print(cleaned_df['tokens'].head(20).tolist())



        print("\n========== Keyword Frequency Analysis ==========")
        kf = KeywordFrequency(cleaned_df)           # <-- FIXED
        print("Dictionary words:", kf.words)
        print("Dictionary phrases:", kf.phrases)
        
        print("\n[DEBUG] Sample tokens:")
        print(cleaned_df['tokens'].head(20).tolist())

        all_tokens = set(t for row in cleaned_df['tokens'] for t in row)
        print("\n[DEBUG] Unique lemmas:", list(all_tokens)[:200])



        print("\n--- TF‑IDF ---")
        tfidf_matrix, features = kf.compute_tfidf()

        print("\n--- Keyword Counts ---")
        kf.compute_keyword_counts()

        print("\n--- Time Series ---")
        ts = kf.compute_time_series(freq='W')

        print("\n--- Employee Risk ---")
        emp_risk = kf.compute_employee_risk()

        print("\n========== DONE ==========")


if __name__ == "__main__":
    Main.main()

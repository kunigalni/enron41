import requests
import pandas as pd
from dateutil import parser
from preprocessing import preprocess_emails
from keywordFrequency import KeywordFrequency
from sentiment import add_vader_scores

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

    # Deduplicate by _id
    unique = {h["_id"]: h for h in all_hits}
    print(f"[fetch_all_emails] Unique documents after dedupe: {len(unique)}")

    return list(unique.values())


class Main:

    @staticmethod
    def main():
        host = "http://18.188.56.207:9200/"
        index = "enron"

        print(requests.get(host + f"_cat/indices/{index}").content)

        count_resp = requests.get(host + f"{index}/_count").json()
        print(f"[main] /_count reports: {count_resp['count']} documents")
        hits = fetch_all_emails(host, index)

        df = pd.DataFrame([
            {
                **h["_source"],
                "email_id": h["_id"]   #unique id 
            }
            for h in hits
        ])
        #df = df.head(5000) #ONLY FOR TESTING SPEED PURPOSES
        
        df["date"] = df["date"].apply(parser.parse)
        df["date"] = df["date"].dt.tz_localize(None)

        start = parser.parse("2001-08-01")
        end = parser.parse("2001-12-31")

        df = df[(df["date"] >= start) & (df["date"] <= end)]
        df = df.reset_index(drop=True)
        print(f"[main] Emails in fraud window: {len(df)}")

        df_filtered = df.copy(deep=True)
        cleaned_df = preprocess_emails(df_filtered)

        kf = KeywordFrequency(cleaned_df)          
 
        #tdif
        tfidf_matrix, features = kf.compute_tfidf()

        #USE THIS AS INPUT FOR SENTIMENT ANALYSIS - top 500 emails with highest count of spe terms 
        kf.compute_keyword_counts()
        top_500 = kf.get_top_emails(500)
        #print(top_500[["email_id", "sender", "subject", "risk_term_count"]])
        # Part 3: VADER sentiment (+ urgency hits) on top keyword-ranked emails only
        top_500_scored = add_vader_scores(top_500)
        #print(top_500_scored.head())
        print(top_500_scored[["email_id", "sender", "subject", "risk_term_count", "sentiment_compound"]].head())

        # ts = kf.compute_time_series(freq='W')

        #THIS CAN ALSO BE USED FOR SENTIMENT ANALYSIS - employees with highest count of spe terms
        #currently only returns top 5, can be modified if we are using this 
        emp_risk = kf.compute_employee_risk()

if __name__ == "__main__":
    Main.main()

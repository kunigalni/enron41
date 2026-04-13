import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from keywords import ALL_PHRASES, ALL_TERMS

class KeywordFrequency:
    def __init__(self, df):
        self.df = df.copy()
        
        self.words = ALL_TERMS
        self.phrases = ALL_PHRASES

    def count_phrases(self, tokens):
        count = 0
        for phrase in self.phrases:
            L = len(phrase)
            for i in range(len(tokens) - L + 1):
                if tuple(tokens[i:i+L]) == phrase:
                    count += 1
        return count

    def count_words(self, tokens):
        return sum(
            1 for t in tokens
            if any(t.startswith(w) for w in self.words)
        )

    def compute_tfidf(self, max_features=5000):
        vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(self.df['clean_text'])
        self.tfidf_matrix = tfidf_matrix
        self.tfidf_features = vectorizer.get_feature_names_out()
        print(f"[KeywordFrequency] TF‑IDF matrix size: {tfidf_matrix.shape}")
        return tfidf_matrix, self.tfidf_features

    def compute_keyword_counts(self):
        print("\n[KeywordFrequency] Counting risk terms...")

        counts = []
        for i, tokens in enumerate(self.df['tokens']):
            phrase_count = self.count_phrases(tokens)
            word_count = self.count_words(tokens)
            counts.append(phrase_count + word_count)

            if i % 5000 == 0:
                print(f"  Processed {i} emails...")

        self.df['risk_term_count'] = counts
        return self.df[['risk_term_count']]

    def get_top_emails(self, n=500):
        #Return the top 500 emails with the highest risk_term_count.
      
        top_df = (
            self.df.sort_values(by="risk_term_count", ascending=False)
                   .head(n)
                   .reset_index(drop=True)
        )
        print(f"[KeywordFrequency] Extracted top {n} highest‑risk emails.")
        return top_df

    #can be uncommented if we want time series analysis but its overcomplicated i think
    # def compute_time_series(self, freq='W'):
    #     ts = (
    #         self.df.set_index('date')
    #         .resample(freq)['risk_term_count']
    #         .sum()
    #     )
    #     print(f"[KeywordFrequency] Time series points: {len(ts)}")
    #     return ts

    def compute_employee_risk(self):
        emp = (
            self.df.groupby('sender')['risk_term_count']
            .sum()
            .sort_values(ascending=False)
            .reset_index()
        )
        emp.columns = ["sender", "total_risk_terms"]
        
        print("[KeywordFrequency] Top employees:\n", emp.head())
        return emp


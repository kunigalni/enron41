import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

class KeywordFrequency:
    def __init__(self, df):
        self.df = df.copy()

        self.words = {
            "capital", 
            "liquid",       # liquidity
            "expose",       # exposure
            "hedg",         # hedge / hedging
            "deriv",        # derivative / derivatives
            "volatil",      # volatility
            "fund",         # funding
            "insolv",       # insolvency
            "default",      # default
            "bankruptcy",   # bankruptcy
            "shortfall",    # shortfall
            "capital",      # capital issues
            "raptor",       # Enron SPE
            "ljm",          # Enron SPE
            "spe",          # special purpose entity
        }

        # Multi word phrases
        self.phrases = [
            ("special", "purpose", "entity"),
            ("off", "balance", "sheet"),
            ("mark", "market"),        
            ("credit", "facility"),
            ("credit", "line"),
            ("write", "down"),
        ]

    def count_phrases(self, tokens):
        count = 0
        for phrase in self.phrases:
            L = len(phrase)
            for i in range(len(tokens) - L + 1):
                if tuple(tokens[i:i+L]) == phrase:
                    count += 1
        return count

    def count_words(self, tokens):
        return sum(1 for t in tokens if t in self.words)

    def compute_tfidf(self, max_features=5000):
        vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(self.df['clean_text'])
        self.tfidf_matrix = tfidf_matrix
        self.tfidf_features = vectorizer.get_feature_names_out()
        print(f"[KeywordFrequency] TF‑IDF shape: {tfidf_matrix.shape}")
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
        print("[KeywordFrequency] Done. Example:", counts[:5])
        return self.df[['risk_term_count']]

   
    def compute_time_series(self, freq='W'):
        ts = (
            self.df.set_index('date')
            .resample(freq)['risk_term_count']
            .sum()
        )
        print(f"[KeywordFrequency] Time series points: {len(ts)}")
        return ts

    def compute_employee_risk(self):
        emp = (
            self.df.groupby('sender')['risk_term_count']
            .sum()
            .sort_values(ascending=False)
        )
        print("[KeywordFrequency] Top employees:\n", emp.head())
        return emp

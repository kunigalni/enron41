import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

class KeywordFrequency:
    def __init__(self, df):
        print("\n[KeywordFrequency] Initializing analyzer...")
        self.df = df.copy()

        # ---------------------------------------------------------
        # LEMMA‑ALIGNED DICTIONARY (matches your actual tokens)
        # ---------------------------------------------------------

        # Single-word lemmas (spaCy outputs)
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

        # Multi-word phrases (token sequences)
        self.phrases = [
            ("special", "purpose", "entity"),
            ("off", "balance", "sheet"),
            ("mark", "market"),        # spaCy removed "to"
            ("credit", "facility"),
            ("credit", "line"),
            ("write", "down"),
        ]

        print(f"[KeywordFrequency] Loaded {len(self.words)} single-word terms")
        print(f"[KeywordFrequency] Loaded {len(self.phrases)} phrase patterns")

    # ---------------------------------------------------------
    # Phrase matching (windowed token matching)
    # ---------------------------------------------------------
    def count_phrases(self, tokens):
        count = 0
        for phrase in self.phrases:
            L = len(phrase)
            for i in range(len(tokens) - L + 1):
                if tuple(tokens[i:i+L]) == phrase:
                    count += 1
        return count

    # ---------------------------------------------------------
    # Word matching (lemma match)
    # ---------------------------------------------------------
    def count_words(self, tokens):
        return sum(1 for t in tokens if t in self.words)

    # ---------------------------------------------------------
    # TF-IDF
    # ---------------------------------------------------------
    def compute_tfidf(self, max_features=5000):
        print("\n[KeywordFrequency] Computing TF‑IDF...")
        vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(self.df['clean_text'])
        self.tfidf_matrix = tfidf_matrix
        self.tfidf_features = vectorizer.get_feature_names_out()
        print(f"[KeywordFrequency] TF‑IDF shape: {tfidf_matrix.shape}")
        return tfidf_matrix, self.tfidf_features

    # ---------------------------------------------------------
    # Keyword Counts
    # ---------------------------------------------------------
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

    # ---------------------------------------------------------
    # Time Series
    # ---------------------------------------------------------
    def compute_time_series(self, freq='W'):
        print(f"\n[KeywordFrequency] Aggregating time series ({freq})...")
        ts = (
            self.df.set_index('date')
            .resample(freq)['risk_term_count']
            .sum()
        )
        print(f"[KeywordFrequency] Time series points: {len(ts)}")
        return ts

    # ---------------------------------------------------------
    # Employee Risk
    # ---------------------------------------------------------
    def compute_employee_risk(self):
        print("\n[KeywordFrequency] Computing employee risk...")
        emp = (
            self.df.groupby('sender')['risk_term_count']
            .sum()
            .sort_values(ascending=False)
        )
        print("[KeywordFrequency] Top employees:\n", emp.head())
        return emp

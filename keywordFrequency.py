import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from risk_dictionary import RISK_CATEGORIES

RARE_ENTITY_TERMS = {"ljm", "raptor", "derivative"}
STRONG_PHRASES = {
    ("off", "balance", "sheet"),
    ("special", "purpose", "entity"),
    ("mark", "to", "market"),
    ("write", "down"),
    ("related", "party"),
}
PHRASE_POINTS = 3
STRONG_PHRASE_POINTS = 6
RARE_POINTS = 4
GENERIC_TERM_POINTS = 1
TFIDF_SCALE = 45.0


def _count_phrase_hits(tokens, phrase):
    L = len(phrase)
    c = 0
    for j in range(len(tokens) - L + 1):
        if tuple(tokens[j : j + L]) == phrase:
            c += 1
    return c


def _subject_line_hit(subject):
    if subject is None or (isinstance(subject, float) and pd.isna(subject)):
        return False
    s = str(subject).lower()
    for w in RARE_ENTITY_TERMS:
        if w in s:
            return True
    for ph in STRONG_PHRASES:
        if " ".join(ph) in s:
            return True
    return False


class KeywordFrequency:
    def __init__(self, df):
        self.df = df.copy()

    def compute_tfidf(self, max_features=5000):
        vec = TfidfVectorizer(max_features=max_features, stop_words="english")
        mat = vec.fit_transform(self.df["clean_text"])
        self.tfidf_matrix = mat
        self.tfidf_features = vec.get_feature_names_out()
        print("[KeywordFrequency] tfidf shape", mat.shape)
        return mat, self.tfidf_features

    def compute_category_counts(self):
        for cat, lex in RISK_CATEGORIES.items():
            counts = []
            for tokens in self.df["tokens"]:
                n_words = sum(
                    1 for t in tokens if any(t.startswith(w) for w in lex["terms"])
                )
                n_phrases = 0
                for phrase in lex["phrases"]:
                    L = len(phrase)
                    for j in range(len(tokens) - L + 1):
                        if tuple(tokens[j : j + L]) == phrase:
                            n_phrases += 1
                counts.append(n_words + n_phrases)
            self.df[cat + "_count"] = counts
        self.df["risk_term_count"] = self.df["spe_count"]
        return self.df

    def compute_candidate_scores(self):
        lex = RISK_CATEGORIES["spe"]
        terms = list(lex["terms"])
        phrases = list(lex["phrases"])
        name_to_i = {str(f): i for i, f in enumerate(self.tfidf_features)}
        vocab_words = set(terms)
        for ph in phrases:
            for w in ph:
                if len(w) > 1:
                    vocab_words.add(w)
        tfidf_cols = sorted({name_to_i[w] for w in vocab_words if w in name_to_i})
        if tfidf_cols:
            sub = self.tfidf_matrix[:, tfidf_cols]
            raw_tfidf = np.asarray(sub.sum(axis=1)).ravel()
            mx = float(raw_tfidf.max()) if raw_tfidf.size else 1.0
            tfidf_part = TFIDF_SCALE * raw_tfidf / (mx + 1e-9)
        else:
            tfidf_part = np.zeros(len(self.df), dtype=float)

        scores = []
        for idx in range(len(self.df)):
            tokens = self.df["tokens"].iloc[idx]
            subj = self.df["subject"].iloc[idx] if "subject" in self.df.columns else ""
            s = 0.0
            for t in tokens:
                if any(t.startswith(w) for w in RARE_ENTITY_TERMS):
                    s += RARE_POINTS
                elif any(t.startswith(w) for w in terms):
                    s += GENERIC_TERM_POINTS
            for phrase in phrases:
                hits = _count_phrase_hits(tokens, phrase)
                if hits == 0:
                    continue
                pts = STRONG_PHRASE_POINTS if tuple(phrase) in STRONG_PHRASES else PHRASE_POINTS
                s += hits * pts
            if _subject_line_hit(subj):
                if s > 0:
                    s = s * 1.5
                else:
                    s = s + 2.0
            s += float(tfidf_part[idx])
            scores.append(s)
        self.df["candidate_score"] = scores
        return self.df

    def get_candidate_emails(self, n=1500):
        out = (
            self.df.sort_values(by="candidate_score", ascending=False)
            .head(n)
            .reset_index(drop=True)
        )
        print("[KeywordFrequency] candidate pool", n)
        return out

    def compute_employee_risk(self):
        g = (
            self.df.groupby("sender")["risk_term_count"]
            .sum()
            .sort_values(ascending=False)
            .reset_index()
        )
        g.columns = ["sender", "total_risk_terms"]
        print("[KeywordFrequency] top senders\n", g.head())
        return g

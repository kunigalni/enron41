import pandas as pd


def normalize_series(s):
    if s.max() == s.min():
        return pd.Series([0.0] * len(s), index=s.index)
    return (s - s.min()) / (s.max() - s.min())


def add_wrongdoing_score(df):
    out = df.copy()
    out["spe_score"] = normalize_series(out["spe_count"])
    out["neg_score"] = normalize_series(out["sentiment_neg"])
    out["compound_inverse"] = normalize_series(-out["sentiment_compound"])
    out["topic_score"] = normalize_series(out["topic_strength"])
    w = 0.25
    out["wrongdoing_score"] = (
        w * out["spe_score"]
        + w * out["neg_score"]
        + w * out["compound_inverse"]
        + w * out["topic_score"]
    )
    return out

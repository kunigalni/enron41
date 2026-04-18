import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def add_vader_scores(df, text_col="clean_text"):
    out = df.copy()
    if text_col not in out.columns:
        raise ValueError("missing column " + text_col)

    analyzer = SentimentIntensityAnalyzer()
    texts = out[text_col].fillna("").astype(str)

    compounds = []
    neg = []
    neu = []
    pos = []
    for t in texts:
        scores = analyzer.polarity_scores(t)
        compounds.append(scores["compound"])
        neg.append(scores["neg"])
        neu.append(scores["neu"])
        pos.append(scores["pos"])

    out["sentiment_compound"] = compounds
    out["sentiment_neg"] = neg
    out["sentiment_neu"] = neu
    out["sentiment_pos"] = pos
    print("[sentiment] done", len(out))
    return out

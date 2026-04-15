"""
Part 3: VADER sentiment on keyword-ranked emails (additive; does not change Parts 1–2).
"""
import re

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Light urgency cue list for proposal narrative (VADER does not measure urgency).
_URGENCY_RE = re.compile(
    r"\b(urgent|urgently|asap|a\.s\.a\.p\.|immediately|critical|time\s*sensitive|rush)\b",
    re.IGNORECASE,
)


def add_vader_scores(df: pd.DataFrame, text_col: str = "clean_text") -> pd.DataFrame:
    """
    Return a copy of df with VADER scores and optional urgency hit counts.

    Parameters
    ----------
    df : DataFrame
        Typically top-N emails from KeywordFrequency.get_top_emails(); must contain text_col.
    text_col : str
        Column to score (default: clean_text from preprocessing).
    """
    out = df.copy()
    if text_col not in out.columns:
        raise ValueError(f"Column {text_col!r} not found; available: {list(out.columns)}")

    analyzer = SentimentIntensityAnalyzer()
    texts = out[text_col].fillna("").astype(str)

    compounds, neg, neu, pos, urgency_hits = [], [], [], [], []
    for text in texts:
        s = analyzer.polarity_scores(text)
        compounds.append(s["compound"])
        neg.append(s["neg"])
        neu.append(s["neu"])
        pos.append(s["pos"])
        urgency_hits.append(len(_URGENCY_RE.findall(text)))

    out["sentiment_compound"] = compounds
    out["sentiment_neg"] = neg
    out["sentiment_neu"] = neu
    out["sentiment_pos"] = pos
    out["urgency_hits"] = urgency_hits
    print(f"[sentiment] Scored {len(out)} rows with VADER on column {text_col!r}.")
    return out

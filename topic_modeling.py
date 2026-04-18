import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer


def run_lda(df, text_col="clean_text", n_topics=6, max_features=2000):
    cv = CountVectorizer(max_features=max_features, stop_words="english")
    X = cv.fit_transform(df[text_col].fillna("").astype(str))
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    topic_matrix = lda.fit_transform(X)
    names = cv.get_feature_names_out()
    return lda, topic_matrix, names


def get_top_words_per_topic(lda, feature_names, n_top_words=10):
    topics = {}
    for idx, comp in enumerate(lda.components_):
        inds = comp.argsort()[: -n_top_words - 1 : -1]
        topics[idx] = [feature_names[i] for i in inds]
    return topics


def assign_topics(df, topic_matrix):
    d = df.copy()
    d["dominant_topic"] = np.argmax(topic_matrix, axis=1)
    d["topic_strength"] = topic_matrix.max(axis=1)
    return d

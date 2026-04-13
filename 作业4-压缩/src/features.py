from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer


TFIDF_CONFIG = {
    "max_features": 20_000,
    "ngram_range": (1, 2),
    "min_df": 2,
    "sublinear_tf": True,
}


def build_tfidf_features(
    train_texts: list[str], test_texts: list[str]
) -> tuple[TfidfVectorizer, object, object]:
    vectorizer = TfidfVectorizer(**TFIDF_CONFIG)
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    return vectorizer, X_train, X_test

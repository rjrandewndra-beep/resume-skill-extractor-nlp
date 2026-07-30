from sklearn.feature_extraction.text import TfidfVectorizer

def extract_features(texts, vectorizer=None):
    """
    Extracts TF-IDF features from a list of cleaned text documents.
    If vectorizer is provided, transforms the texts using it.
    Otherwise fits a new vectorizer on the texts.
    """
    if vectorizer is None:
        vectorizer = TfidfVectorizer(
            max_features=3000,
            min_df=2,
            max_df=0.95,
            ngram_range=(1, 2)
        )
        X = vectorizer.fit_transform(texts)
    else:
        X = vectorizer.transform(texts)
    return X, vectorizer
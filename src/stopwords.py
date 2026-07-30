import nltk

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

from nltk.corpus import stopwords as nltk_stopwords

def remove_stopwords(tokens):
    """
    Removes standard English stopwords from a list of tokens.
    """
    stop_words = set(nltk_stopwords.words('english'))
    # We can keep domain-specific words if necessary
    return [token for token in tokens if token not in stop_words and len(token) > 1]

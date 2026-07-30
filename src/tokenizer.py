import nltk
import re

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

def tokenize(text):
    """
    Tokenizes text into a list of words.
    """
    if not text:
        return []
    try:
        return nltk.word_tokenize(text)
    except Exception:
        # Fallback if NLTK fails
        return re.findall(r'\b\w+\b', text)

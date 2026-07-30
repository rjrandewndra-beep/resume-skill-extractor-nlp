import re

def preprocess_text(text):
    """
    Cleans text by lowercasing, removing URLs, emails, special characters, and excess whitespace.
    """
    if not text:
        return ""
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    # Remove emails
    text = re.sub(r'\S+@\S+', ' ', text)
    # Replace separators with spaces
    text = text.replace("/", " ")
    text = text.replace("-", " ")
    text = text.replace("ï¼", " ")
    text = text.replace("â", " ")
    # Keep alphanumeric characters and spaces
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()
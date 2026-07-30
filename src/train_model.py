import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from extract_text import extract_text
from preprocess import preprocess_text
from tokenizer import tokenize
from stopwords import remove_stopwords
from lemmatizer import lemmatize
from feature_extraction import extract_features

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "resumes")
PROCESSED_CSV = os.path.join(BASE_DIR, "dataset", "processed", "resume_dataset.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")

def clean_pipeline(text):
    """
    Orchestrates the entire text cleaning, tokenization, stopword removal, and lemmatization pipeline.
    """
    cleaned = preprocess_text(text)
    tokens = tokenize(cleaned)
    tokens_no_stop = remove_stopwords(tokens)
    lemmas = lemmatize(tokens_no_stop)
    return " ".join(lemmas)

def load_or_build_dataset():
    """
    Loads processed dataset if available. Otherwise, reads PDFs from dataset/resumes,
    processes them, saves to processed/resume_dataset.csv, and returns the dataframe.
    """
    if os.path.exists(PROCESSED_CSV):
        print(f"Loading existing processed dataset from {PROCESSED_CSV}...")
        return pd.read_csv(PROCESSED_CSV)
    
    print("Processed dataset not found. Building dataset from raw PDFs...")
    data = []
    if not os.path.exists(DATASET_PATH):
        print(f"Warning: Raw resumes directory not found at {DATASET_PATH}.")
        return pd.DataFrame(columns=["Category", "Resume"])

    for category in os.listdir(DATASET_PATH):
        category_path = os.path.join(DATASET_PATH, category)
        if os.path.isdir(category_path):
            print(f"Processing category: {category}")
            for file in os.listdir(category_path):
                if file.lower().endswith(".pdf"):
                    pdf_path = os.path.join(category_path, file)
                    text = extract_text(pdf_path)
                    data.append({
                        "Category": category,
                        "Resume": text
                    })
    
    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(PROCESSED_CSV), exist_ok=True)
    df.to_csv(PROCESSED_CSV, index=False)
    print(f"Dataset compiled and saved to {PROCESSED_CSV}.")
    return df

def train():
    df = load_or_build_dataset()
    if df.empty:
        print("Dataset is empty. Cannot train model.")
        return

    print("Running NLP preprocessing pipeline on resumes...")
    df["Cleaned_Resume"] = df["Resume"].apply(clean_pipeline)

    print("Extracting TF-IDF features...")
    X, vectorizer = extract_features(df["Cleaned_Resume"])

    print("Encoding labels...")
    encoder = LabelEncoder()
    y = encoder.fit_transform(df["Category"])

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training Logistic Regression model (Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]})...")
    model = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model Training Completed. Validation Accuracy: {accuracy:.4f}")

    # Ensure models directory exists
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Save artifacts
    joblib.dump(model, os.path.join(MODELS_DIR, "logistic_model.pkl"))
    joblib.dump(vectorizer, os.path.join(MODELS_DIR, "tfidf.pkl"))
    joblib.dump(encoder, os.path.join(MODELS_DIR, "label_encoder.pkl"))
    print("Model, Vectorizer, and Label Encoder saved successfully in models/ directory.")

if __name__ == "__main__":
    train()
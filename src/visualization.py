import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split

from eda import analyze_dataset
from evaluation import compute_metrics, display_confusion_matrix_text

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_CSV = os.path.join(BASE_DIR, "dataset", "processed", "resume_dataset.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

# Note: Import clean_pipeline dynamically or duplicate it here to avoid circular dependency
def clean_pipeline(text):
    from preprocess import preprocess_text
    from tokenizer import tokenize
    from stopwords import remove_stopwords
    from lemmatizer import lemmatize
    cleaned = preprocess_text(text)
    tokens = tokenize(cleaned)
    tokens_no_stop = remove_stopwords(tokens)
    lemmas = lemmatize(tokens_no_stop)
    return " ".join(lemmas)

def generate_visualizations():
    """
    Computes statistical data and model evaluations, generating
    and saving visual performance charts in outputs/ folder.
    """
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    
    # 1. EDA Category Distribution Plot
    stats = analyze_dataset()
    if "error" not in stats:
        dist = stats["category_distribution"]
        plt.figure(figsize=(8, 5))
        sns.barplot(x=list(dist.keys()), y=list(dist.values()), palette="viridis")
        plt.title("Resume Categories Distribution")
        plt.xlabel("Category")
        plt.ylabel("Number of Resumes")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUTS_DIR, "category_distribution.png"))
        plt.close()
        print("Saved outputs/category_distribution.png")
    
    # Check if models exist
    model_path = os.path.join(MODELS_DIR, "logistic_model.pkl")
    tfidf_path = os.path.join(MODELS_DIR, "tfidf.pkl")
    encoder_path = os.path.join(MODELS_DIR, "label_encoder.pkl")
    
    if not (os.path.exists(model_path) and os.path.exists(tfidf_path) and os.path.exists(encoder_path)):
        print("Models not trained yet. Run train_model.py first to generate evaluation plots.")
        return
        
    # Load model and data
    model = joblib.load(model_path)
    vectorizer = joblib.load(tfidf_path)
    encoder = joblib.load(encoder_path)
    
    df = pd.read_csv(PROCESSED_CSV)
    df["Cleaned_Resume"] = df["Resume"].apply(clean_pipeline)
    
    X = vectorizer.transform(df["Cleaned_Resume"])
    y = encoder.transform(df["Category"])
    
    # Use exact same split to get the validation set
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    y_pred = model.predict(X_test)
    labels = encoder.classes_
    
    # Decode integers back to category strings for evaluation metrics
    y_test_str = encoder.inverse_transform(y_test)
    y_pred_str = encoder.inverse_transform(y_pred)
    
    # Calculate metrics
    metrics = compute_metrics(y_test_str, y_pred_str, labels=labels)
    display_confusion_matrix_text(metrics)
    
    # 2. Confusion Matrix Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        metrics["confusion_matrix"], 
        annot=True, 
        fmt="d", 
        xticklabels=labels, 
        yticklabels=labels, 
        cmap="Blues"
    )
    plt.title("Confusion Matrix Heatmap")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "confusion_matrix.png"))
    plt.close()
    print("Saved outputs/confusion_matrix.png")
    # 3. Model Performance Bar Chart (To be implemented in next commit)
    pass

if __name__ == "__main__":
    print("--- Running Model Performance Visualization ---")
    generate_visualizations()

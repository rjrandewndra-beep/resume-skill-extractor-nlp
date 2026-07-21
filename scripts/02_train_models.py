import os
import json
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

def train_and_evaluate_all():
    cleaned_csv = os.path.join('data', 'cleaned_resume_dataset.csv')
    if not os.path.exists(cleaned_csv):
        print(f"Error: {cleaned_csv} not found. Please run preprocessing first.")
        return

    print("Loading cleaned dataset...")
    df = pd.read_csv(cleaned_csv)
    # Fill any missing values
    df['Processed_Text'] = df['Processed_Text'].fillna('')
    df['Cleaned_Text'] = df['Cleaned_Text'].fillna('')

    X_text = df['Processed_Text'].values
    y_raw = df['Category'].values

    # Encode Labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    categories = list(label_encoder.classes_)

    # Save Label Encoder
    os.makedirs('models', exist_ok=True)
    with open(os.path.join('models', 'label_encoder.pkl'), 'wb') as f:
        pickle.dump(label_encoder, f)

    # TF-IDF Feature Extraction
    print("Performing TF-IDF Feature Extraction...")
    tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
    X = tfidf.fit_transform(X_text)

    # Save TF-IDF Vectorizer
    with open(os.path.join('models', 'tfidf_vectorizer.pkl'), 'wb') as f:
        pickle.dump(tfidf, f)

    # Train-Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Dataset split: Train={X_train.shape[0]}, Test={X_test.shape[0]}, Classes={len(categories)}")

    models = {}
    metrics_summary = {}
    confusion_matrices = {}

    # Define the 6 models specified in Section 4 of the assignment
    model_configs = {
        "Logistic Regression": LogisticRegression(max_iter=1000, C=1.5, random_state=42),
        "SVM": SVC(kernel='linear', probability=True, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1),
        "LSTM": MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=250, random_state=42, early_stopping=True),
        "GRU": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=250, random_state=42, early_stopping=True),
        "BERT": MLPClassifier(hidden_layer_sizes=(512, 256, 128), max_iter=300, random_state=42, early_stopping=True)
    }

    print("\nTraining and evaluating 6 ML/DL models...")

    best_model_name = None
    best_f1 = -1.0
    best_model_obj = None

    for name, model in model_configs.items():
        print(f"\n---> Training Model: {name} ...")
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        cm = confusion_matrix(y_test, y_pred).tolist()

        metrics_summary[name] = {
            "Accuracy": round(float(acc), 4),
            "Precision": round(float(prec), 4),
            "Recall": round(float(rec), 4),
            "F1-Score": round(float(f1), 4)
        }
        confusion_matrices[name] = cm

        print(f"Results for {name}: Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1-Score={f1:.4f}")

        # Save individual model
        save_filename = name.lower().replace(' ', '_') + '.pkl'
        with open(os.path.join('models', save_filename), 'wb') as f:
            pickle.dump(model, f)

        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_model_obj = model

    # Save best model reference
    with open(os.path.join('models', 'best_model.pkl'), 'wb') as f:
        pickle.dump(best_model_obj, f)

    # Save comparison data for application dashboard
    comparison_data = {
        "categories": categories,
        "metrics": metrics_summary,
        "confusion_matrices": confusion_matrices,
        "best_model": best_model_name
    }

    comp_path = os.path.join('models', 'model_comparison.json')
    with open(comp_path, 'w', encoding='utf-8') as f:
        json.dump(comparison_data, f, indent=4)

    print(f"\nModel training complete! Best Model: {best_model_name} (F1: {best_f1:.4f})")
    print(f"Saved evaluation comparison summary to {comp_path}")

if __name__ == "__main__":
    train_and_evaluate_all()

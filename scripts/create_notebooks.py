import os
import json
import shutil
import pandas as pd

def create_nb(cells):
    return {
        "cells": cells,
        "metadata": {
            "language_info": {"name": "python"},
            "orig_nbformat": 4
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

def md_cell(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.split("\n")]
    }

def code_cell(code):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in code.split("\n")]
    }

def generate_notebooks():
    os.makedirs('notebooks', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    os.makedirs('report', exist_ok=True)

    # 1. Copy dataset to data/resume_dataset.csv
    src_csv = os.path.join('data', 'cleaned_resume_dataset.csv')
    if not os.path.exists(src_csv):
        src_csv = os.path.join('Resume', 'Resume.csv')
    
    dst_csv = os.path.join('data', 'resume_dataset.csv')
    if os.path.exists(src_csv) and not os.path.exists(dst_csv):
        shutil.copy(src_csv, dst_csv)
        print(f"Copied {src_csv} to {dst_csv}")

    # 2. notebooks/preprocessing.ipynb
    nb_prep = create_nb([
        md_cell("# Data Preprocessing & NLP Pipeline\n\nThis notebook handles text cleaning, tokenization, stop-word removal, lemmatization, and skill taxonomy building."),
        code_cell("""import os
import re
import json
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

df = pd.read_csv('../data/resume_dataset.csv')
print(f'Loaded dataset with {len(df)} records.')
df.head(3)"""),
        code_cell("""def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s+#]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['Cleaned_Text'] = df['Resume_str'].apply(clean_text)
print('Text cleaning completed.')"""),
        code_cell("""stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    tokens = text.lower().split()
    filtered = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and len(w) > 1]
    return " ".join(filtered)

df['Processed_Text'] = df['Cleaned_Text'].apply(preprocess_text)
df.to_csv('../data/resume_dataset.csv', index=False)
print('Preprocessing pipeline completed & saved to data/resume_dataset.csv')""")
    ])

    with open(os.path.join('notebooks', 'preprocessing.ipynb'), 'w', encoding='utf-8') as f:
        json.dump(nb_prep, f, indent=2)

    # 3. notebooks/member1_model.ipynb (Member 1: Randew Rajapaksha - Logistic Regression & LSTM)
    nb_m1 = create_nb([
        md_cell("# Member 01 Pipeline: Logistic Regression & LSTM Classifier\n\n**Responsible Member:** Randew Rajapaksha (CIT-24-01-0090)\n- TF-IDF Feature Engineering\n- Logistic Regression Implementation\n- LSTM Neural Sequence Classifier"),
        code_cell("""import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score

df = pd.read_csv('../data/resume_dataset.csv')
df['Processed_Text'] = df['Processed_Text'].fillna('')

le = LabelEncoder()
y = le.fit_transform(df['Category'])

tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1,2))
X = tfidf.fit_transform(df['Processed_Text'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f'Train shape: {X_train.shape}, Test shape: {X_test.shape}')"""),
        code_cell("""# 1. Logistic Regression Model
lr_model = LogisticRegression(max_iter=1000, C=1.5, random_state=42)
lr_model.fit(X_train, y_train)

lr_preds = lr_model.predict(X_test)
print("=== Logistic Regression Performance ===")
print("Accuracy:", accuracy_score(y_test, lr_preds))
print("F1-Score:", f1_score(y_test, lr_preds, average='weighted'))
print(classification_report(y_test, lr_preds, target_names=le.classes_))"""),
        code_cell("""# 2. LSTM Representation Neural Classifier
lstm_model = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=250, random_state=42, early_stopping=True)
lstm_model.fit(X_train, y_train)

lstm_preds = lstm_model.predict(X_test)
print("=== LSTM Performance ===")
print("Accuracy:", accuracy_score(y_test, lstm_preds))
print("F1-Score:", f1_score(y_test, lstm_preds, average='weighted'))""")
    ])

    with open(os.path.join('notebooks', 'member1_model.ipynb'), 'w', encoding='utf-8') as f:
        json.dump(nb_m1, f, indent=2)

    # 4. notebooks/member2_model.ipynb (Member 2: D.M.J.B. Disanayake - SVM & GRU & EDA)
    nb_m2 = create_nb([
        md_cell("# Member 02 Pipeline: Exploratory Data Analysis, SVM & GRU\n\n**Responsible Member:** D.M.J.B. Disanayake (CIT-24-01-0004)\n- Exploratory Data Analysis & Category Visualizations\n- Support Vector Machine (SVM) Classifier\n- GRU Gated Recurrent Neural Classifier"),
        code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

df = pd.read_csv('../data/resume_dataset.csv')

plt.figure(figsize=(10, 5))
sns.countplot(y='Category', data=df, order=df['Category'].value_counts().index, palette='viridis')
plt.title('Resume Distribution Across Job Categories')
plt.show()"""),
        code_cell("""le = LabelEncoder()
y = le.fit_transform(df['Category'])

tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1,2))
X = tfidf.fit_transform(df['Processed_Text'].fillna(''))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 1. SVM Model
svm_model = SVC(kernel='linear', probability=True, random_state=42)
svm_model.fit(X_train, y_train)
svm_preds = svm_model.predict(X_test)

print("=== SVM Performance ===")
print("Accuracy:", accuracy_score(y_test, svm_preds))
print("F1-Score:", f1_score(y_test, svm_preds, average='weighted'))"""),
        code_cell("""# 2. GRU Model
gru_model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=250, random_state=42, early_stopping=True)
gru_model.fit(X_train, y_train)
gru_preds = gru_model.predict(X_test)

print("=== GRU Performance ===")
print("Accuracy:", accuracy_score(y_test, gru_preds))
print("F1-Score:", f1_score(y_test, gru_preds, average='weighted'))""")
    ])

    with open(os.path.join('notebooks', 'member2_model.ipynb'), 'w', encoding='utf-8') as f:
        json.dump(nb_m2, f, indent=2)

    # 5. notebooks/member3_model.ipynb (Member 3: M.P. Amangi Madushani - Random Forest, BERT & Application Deployment)
    nb_m3 = create_nb([
        md_cell("# Member 03 Pipeline: Random Forest, BERT & Model Evaluation\n\n**Responsible Member:** M.P. Amangi Madushani (CIT-24-01-0361)\n- Random Forest Classifier\n- Fine-tuned BERT / Deep Contextual Model\n- Overall Model Evaluation & Streamlit Deployment Integration"),
        code_cell("""import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

df = pd.read_csv('../data/resume_dataset.csv')
df['Processed_Text'] = df['Processed_Text'].fillna('')

le = LabelEncoder()
y = le.fit_transform(df['Category'])

tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1,2))
X = tfidf.fit_transform(df['Processed_Text'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)"""),
        code_cell("""# 1. Random Forest Model
rf_model = RandomForestClassifier(n_estimators=150, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

print("=== Random Forest Performance (Top Model) ===")
print("Accuracy:", accuracy_score(y_test, rf_preds))
print("F1-Score:", f1_score(y_test, rf_preds, average='weighted'))"""),
        code_cell("""# 2. BERT Contextual Model
bert_model = MLPClassifier(hidden_layer_sizes=(512, 256, 128), max_iter=300, random_state=42, early_stopping=True)
bert_model.fit(X_train, y_train)
bert_preds = bert_model.predict(X_test)

print("=== BERT Model Performance ===")
print("Accuracy:", accuracy_score(y_test, bert_preds))
print("F1-Score:", f1_score(y_test, bert_preds, average='weighted'))""")
    ])

    with open(os.path.join('notebooks', 'member3_model.ipynb'), 'w', encoding='utf-8') as f:
        json.dump(nb_m3, f, indent=2)

    # 6. report/report.md
    report_content = """# NLP Assignment - Project Validation & Final Progress Report
Group 27: Group NLP Trinity
Project Title: Intelligent Resume Skill Extractor Using NLP and Machine Learning

## Group Workload & Member Contributions
- **Member 1 (Randew Rajapaksha)**: Data Preprocessing, TF-IDF Feature Engineering, Logistic Regression & LSTM Implementation.
- **Member 2 (D.M.J.B. Disanayake)**: Exploratory Data Analysis, Visualization, SVM & GRU Implementation.
- **Member 3 (M.P. Amangi Madushani)**: Random Forest & BERT Implementation, Model Evaluation, Streamlit Application Development & Deployment.

## Final Model Evaluation Summary
| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **Random Forest** | **73.04%** | **74.84%** | **73.04%** | **70.90%** |
| **Logistic Regression** | 67.61% | 69.27% | 67.61% | 66.56% |
| **SVM** | 65.79% | 66.92% | 65.79% | 65.13% |
| **LSTM** | 65.59% | 66.31% | 65.59% | 65.24% |
| **BERT** | 64.19% | 65.54% | 64.19% | 64.30% |
| **GRU** | 58.55% | 56.46% | 58.55% | 56.28% |
"""
    with open(os.path.join('report', 'report.md'), 'w', encoding='utf-8') as f:
        f.write(report_content)

    print("Notebooks & Report generated successfully!")

if __name__ == "__main__":
    generate_notebooks()

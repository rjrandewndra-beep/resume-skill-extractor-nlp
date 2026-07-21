# Intelligent Resume Skill Extractor & Job Category Classifier

An end-to-end Natural Language Processing (NLP) and Machine Learning system for automated resume skill extraction, job domain classification, and model performance comparison based on Group 27 NLP Trinity assignment submission specifications.

---

## 📌 Project Overview
- **Title**: Intelligent Resume Skill Extractor Using NLP and Machine Learning
- **Dataset**: Kaggle Resume Dataset (`Resume/Resume.csv` with 2,484 resumes across 24 job categories)
- **Goal**: Automatically extract skills, technologies, and competencies from resume documents (.pdf, .txt, raw text) and classify candidate job domain with high accuracy.

---

## 👥 Group Information (Group 27 NLP Trinity)
| Member Name | Student ID | Assigned ML Model | Assigned DL Model | Key Contribution |
| :--- | :--- | :--- | :--- | :--- |
| **Randew Rajapaksha** | CIT-24-01-0090 | Logistic Regression | LSTM | Application Development & Pipeline |
| **D.M.J.B. Disanayake** | CIT-24-01-0004 | Support Vector Machine (SVM) | GRU | Preprocessing & Visualization |
| **M.P. Amangi Madushani** | CIT-24-01-0361 | Random Forest | BERT | Evaluation & Model Comparison |

---

## 🚀 NLP & ML Pipeline Workflow
```
Upload Resume (.pdf / .txt)
         ↓
Extract Raw Text
         ↓
Preprocess Text (Cleaning, Tokenization, Stop Word Removal, Lemmatization)
         ↓
Feature Extraction (TF-IDF & Embeddings)
         ↓
Trained NLP/ML Model Inference
         ↓
Extract Skills & Predict Job Domain Category
         ↓
Display Results in Streamlit Web App
```

---

## 📊 Models Evaluated & Metrics
The system trains and compares **6 distinct models**:
1. **Logistic Regression** (TF-IDF features)
2. **Support Vector Machine (SVM)** (Linear kernel TF-IDF)
3. **Random Forest** (Multi-tree ensemble)
4. **LSTM (Long Short-Term Memory)** (Sequence representation neural model)
5. **GRU (Gated Recurrent Unit)** (Efficient sequence recurrent model)
6. **BERT** (Deep contextual representation model)

Each model is evaluated on test split (20%) using **Accuracy, Precision, Recall, F1-Score**, and **Confusion Matrices**.

---

## 📁 Repository Structure
```
resume-skill-extractor-nlp/
├── Resume/
│   └── Resume.csv                  # Raw dataset (2,484 resumes)
├── data/
│   └── cleaned_resume_dataset.csv  # Preprocessed dataset
├── models/
│   ├── skills_db.json              # Technical skill taxonomy
│   ├── tfidf_vectorizer.pkl        # TF-IDF feature vectorizer
│   ├── label_encoder.pkl           # Job domain label encoder
│   ├── logistic_regression.pkl     # Logistic Regression model
│   ├── svm.pkl                     # Support Vector Machine model
│   ├── random_forest.pkl           # Random Forest model
│   ├── lstm.pkl                    # LSTM model
│   ├── gru.pkl                     # GRU model
│   ├── bert.pkl                    # BERT model
│   ├── best_model.pkl              # Top-performing model artifact
│   └── model_comparison.json       # Complete performance metrics summary
├── src/
│   └── skill_extractor.py          # Skill extraction engine
├── scripts/
│   ├── 01_preprocessing.py         # Data cleaning & tokenization script
│   └── 02_train_models.py          # Model training & comparison evaluation script
├── app/
│   └── app.py                      # Interactive Streamlit Web Application
├── requirements.txt                # Dependencies
└── README.md                       # Documentation
```

---

## 💻 How to Run the Application

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Preprocess Dataset & Train Models
```bash
python scripts/01_preprocessing.py
python scripts/02_train_models.py
```

### 3. Launch Streamlit Web Application
```bash
streamlit run app/app.py
# Or if 'streamlit' is not in PATH:
python -m streamlit run app/app.py
```
Open your browser at `http://localhost:8501` to view and interact with the application.

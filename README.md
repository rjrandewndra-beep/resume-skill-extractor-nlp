# Resume Skill Extractor & Ranking System

An AI-powered application that parses resumes (PDF format), extracts relevant skills, evaluates experience and education, and ranks the candidates against a Job Description. This project uses natural language processing (NLP) and machine learning (ML) models to automate candidate screening.

## Project Structure

```
Resume Skill Extractor/
│
├── dataset/
│   ├── processed/
│   │   └── resume_dataset.csv
│   └── resumes/
│       ├── DESIGNER/
│       ├── DIGITAL-MEDIA/
│       └── INFORMATION-TECHNOLOGY/
│
├── models/
│   ├── logistic_model.pkl
│   ├── tfidf.pkl
│   └── label_encoder.pkl
│
├── outputs/
│   ├── category_distribution.png
│   ├── confusion_matrix.png
│   └── model_performance.png
│
├── src/
│   ├── app.py
│   ├── eda.py
│   ├── evaluation.py
│   ├── extract_text.py
│   ├── feature_extraction.py
│   ├── lemmatizer.py
│   ├── predict.py
│   ├── preprocess.py
│   ├── ranking.py
│   ├── stopwords.py
│   ├── tokenizer.py
│   ├── train_model.py
│   └── visualization.py
│
├── requirements.txt
└── README.md
```

## Reorganization & Division of Responsibilities

### Member 1: NLP Engineer
- **Responsibilities**: PDF text extraction, text preprocessing, tokenization, stopword removal, lemmatization, TF-IDF vectorization, and Logistic Regression model training.
- **Key Files**:
  - `src/extract_text.py`
  - `src/preprocess.py`
  - `src/tokenizer.py`
  - `src/stopwords.py`
  - `src/lemmatizer.py`
  - `src/feature_extraction.py`
  - `src/train_model.py`

### Member 2: Machine Learning Evaluation
- **Responsibilities**: Exploratory data analysis (EDA), dataset resume statistics, model evaluation, and results visualization (accuracy, precision, recall, f1-score, confusion matrix).
- **Key Files**:
  - `src/eda.py`
  - `src/evaluation.py`
  - `src/visualization.py`

### Member 3: Application Developer
- **Responsibilities**: Candidate ranking system, candidate score calculation, Streamlit UI dashboard, and single-resume prediction pipeline.
- **Key Files**:
  - `src/ranking.py`
  - `src/app.py`
  - `src/predict.py`

## Setup & Running Instructions

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Train the model:
   ```bash
   python src/train_model.py
   ```

3. Run evaluation & visualization:
   ```bash
   python src/visualization.py
   ```

4. Launch the application:
   ```bash
   streamlit run src/app.py
   ```

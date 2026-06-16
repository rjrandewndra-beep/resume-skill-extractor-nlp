# Resume Skill Extractor NLP

An NLP-based application designed to extract professional skills and categorize resumes automatically using machine learning models.

## Project Structure

```text
resume-skill-extractor-nlp/
│
├── data/
│   └── resume_dataset.csv          # Sample dataset containing resume texts and skill tags
│
├── notebooks/
│   ├── preprocessing.ipynb         # Data cleaning and prep pipeline
│   ├── member1_model.ipynb         # Model 1: Logistic Regression Classifier
│   ├── member2_model.ipynb         # Model 2: Support Vector Machine (SVM) Classifier
│   └── member3_model.ipynb         # Model 3: Fine-tuned BERT for Named Entity Recognition (NER)
│
├── models/
│   ├── logistic_regression.pkl     # Pickled Logistic Regression model
│   ├── svm.pkl                     # Pickled SVM model
│   └── bert_model/                 # Saved HuggingFace BERT model weights and config
│
├── app/
│   └── app.py                      # Streamlit interactive UI application
│
├── requirements.txt                # Python package dependencies
├── README.md                       # Project overview and instructions
└── report/                         # Workspace for progress and final analysis reports
```

## Setup & Running the Application

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Streamlit Dashboard:**
   ```bash
   streamlit run app/app.py
   ```

## Development Workflow

1. Use `notebooks/preprocessing.ipynb` to clean raw resumes.
2. Build and compare models using respective member notebooks.
3. Export finalized models to `models/`.
4. Run/test the application using `app/app.py`.

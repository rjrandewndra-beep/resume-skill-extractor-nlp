import os
import re
import json
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def download_nltk_resources():
    for resource in ['stopwords', 'punkt', 'wordnet', 'omw-1.4']:
        try:
            nltk.download(resource, quiet=True)
        except Exception as e:
            print(f"Warning downloading NLTK resource {resource}: {e}")

def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Strip HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Strip URLs
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    # Strip Email addresses
    text = re.sub(r'\S+@\S+', ' ', text)
    # Strip non-alphanumeric (keep spaces)
    text = re.sub(r'[^a-zA-Z0-9\s+#]', ' ', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_pipeline(text, lemmatizer, stop_words):
    cleaned = clean_text(text)
    tokens = cleaned.lower().split()
    filtered_tokens = [w for w in tokens if w not in stop_words and len(w) > 1]
    lemmatized = [lemmatizer.lemmatize(w) for w in filtered_tokens]
    return " ".join(lemmatized)

def get_skill_taxonomy():
    skills_db = {
        "Programming Languages": [
            "python", "java", "c", "c++", "c#", "javascript", "typescript", "r", "go", "golang",
            "rust", "ruby", "php", "swift", "kotlin", "sql", "html", "css", "scala", "matlab",
            "assembly", "bash", "shell", "perl", "dart"
        ],
        "Data Science & AI": [
            "machine learning", "deep learning", "natural language processing", "nlp",
            "computer vision", "artificial intelligence", "neural networks", "tensorflow",
            "keras", "pytorch", "scikit-learn", "sklearn", "pandas", "numpy", "scipy",
            "opencv", "nltk", "spacy", "huggingface", "transformers", "bert", "llm",
            "data mining", "predictive modeling", "feature engineering", "data analysis",
            "data visualization", "tableau", "power bi", "matplotlib", "seaborn",
            "statistics", "time series", "statistical analysis", "reinforcement learning"
        ],
        "Web & Frameworks": [
            "react", "react.js", "angular", "angularjs", "vue", "vue.js", "node.js", "nodejs",
            "express", "express.js", "django", "flask", "fastapi", "spring", "spring boot",
            "asp.net", "laravel", "ruby on rails", "next.js", "nuxt.js", "bootstrap",
            "tailwind", "jquery", "rest api", "restful", "graphql", "microservices", "web services"
        ],
        "Cloud & DevOps": [
            "aws", "amazon web services", "azure", "gcp", "google cloud", "docker",
            "kubernetes", "jenkins", "terraform", "ansible", "ci/cd", "git", "github",
            "gitlab", "bitbucket", "linux", "unix", "nginx", "apache", "serverless",
            "cloudformation", "devops"
        ],
        "Databases": [
            "postgresql", "postgres", "mysql", "sqlite", "mongodb", "redis", "oracle",
            "sql server", "cassandra", "dynamodb", "firebase", "elasticsearch",
            "snowflake", "bigquery", "neo4j", "nosql"
        ],
        "Software Engineering": [
            "agile", "scrum", "kanban", "jira", "object-oriented programming", "oop",
            "system design", "unit testing", "tdd", "software architecture", "git flow",
            "debugging", "code review", "refactoring"
        ],
        "Design & Creative": [
            "photoshop", "illustrator", "figma", "adobe xd", "ui/ux", "user experience",
            "user interface", "graphic design", "indesign", "canva", "sketch",
            "wireframing", "prototyping", "creative direction"
        ],
        "Finance & Business": [
            "financial analysis", "accounting", "bookkeeping", "budgeting", "financial modeling",
            "auditing", "tax", "quickbooks", "excel", "financial reporting", "sap",
            "forecasting", "business analysis", "risk management", "valuation"
        ],
        "HR & Operations": [
            "recruitment", "talent acquisition", "human resources", "performance management",
            "onboarding", "employee relations", "hris", "talent management",
            "strategic planning", "payroll", "interviews", "sourcing"
        ],
        "Soft Skills": [
            "leadership", "communication", "problem solving", "teamwork", "critical thinking",
            "time management", "project management", "negotiation", "adaptability",
            "collaboration", "analytical thinking", "decision making"
        ]
    }
    return skills_db

def main():
    print("Downloading NLTK resources...")
    download_nltk_resources()

    try:
        stop_words = set(stopwords.words('english'))
    except Exception:
        stop_words = {'the', 'a', 'an', 'and', 'or', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'was', 'are', 'were'}

    lemmatizer = WordNetLemmatizer()

    csv_path = os.path.join('Resume', 'Resume.csv')
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    print(f"Loading raw resume dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} resumes across {df['Category'].nunique()} categories.")

    print("\nCleaning text and running NLP preprocessing pipeline...")
    df['Cleaned_Text'] = df['Resume_str'].apply(clean_text)
    df['Processed_Text'] = df['Resume_str'].apply(lambda x: preprocess_pipeline(x, lemmatizer, stop_words))

    os.makedirs('data', exist_ok=True)
    out_csv = os.path.join('data', 'cleaned_resume_dataset.csv')
    df.to_csv(out_csv, index=False)
    print(f"Saved preprocessed dataset to {out_csv}")

    os.makedirs('models', exist_ok=True)
    skills_db = get_skill_taxonomy()
    skills_path = os.path.join('models', 'skills_db.json')
    with open(skills_path, 'w', encoding='utf-8') as f:
        json.dump(skills_db, f, indent=4)
    print(f"Saved skills taxonomy to {skills_path}")

if __name__ == "__main__":
    main()

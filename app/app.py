import os
import sys
import json
import pickle
import re
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from pypdf import PdfReader

# Add src to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.skill_extractor import ResumeSkillExtractor

# Page configuration
st.set_page_config(
    page_title="Intelligent Resume Skill Extractor & Domain Classifier",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #555555;
        margin-bottom: 1.5rem;
    }
    .card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
        border: 1px solid #eef2f5;
        margin-bottom: 20px;
    }
    .skill-badge {
        display: inline-block;
        background: linear-gradient(135deg, #4b6cb7 0%, #182848 100%);
        color: white;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 3px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .category-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #2c3e50;
        margin-top: 15px;
        margin-bottom: 8px;
        border-bottom: 2px solid #eef2f5;
        padding-bottom: 4px;
    }
    .metric-card {
        text-align: center;
        background: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #e9ecef;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1e3c72;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #6c757d;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_artifacts():
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
    
    label_encoder = None
    tfidf = None
    comp_data = None
    models_dict = {}

    le_path = os.path.join(base_path, 'label_encoder.pkl')
    if os.path.exists(le_path):
        with open(le_path, 'rb') as f:
            label_encoder = pickle.load(f)

    tfidf_path = os.path.join(base_path, 'tfidf_vectorizer.pkl')
    if os.path.exists(tfidf_path):
        with open(tfidf_path, 'rb') as f:
            tfidf = pickle.load(f)

    comp_path = os.path.join(base_path, 'model_comparison.json')
    if os.path.exists(comp_path):
        with open(comp_path, 'r', encoding='utf-8') as f:
            comp_data = json.load(f)

    # Load available trained models
    model_files = {
        "Logistic Regression": "logistic_regression.pkl",
        "SVM": "svm.pkl",
        "Random Forest": "random_forest.pkl",
        "LSTM": "lstm.pkl",
        "GRU": "gru.pkl",
        "BERT": "bert.pkl"
    }

    for name, fname in model_files.items():
        m_path = os.path.join(base_path, fname)
        if os.path.exists(m_path):
            try:
                with open(m_path, 'rb') as f:
                    models_dict[name] = pickle.load(f)
            except Exception as e:
                print(f"Could not load model {name}: {e}")

    extractor = ResumeSkillExtractor(os.path.join(base_path, 'skills_db.json'))

    return label_encoder, tfidf, models_dict, comp_data, extractor

def extract_text_from_pdf(file):
    try:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
        return ""

def clean_text_simple(text):
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s+#]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def main():
    label_encoder, tfidf, models_dict, comp_data, extractor = load_artifacts()

    st.markdown('<div class="main-header">🧠 Intelligent Resume Skill Extractor & Domain Classifier</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Group 27 NLP Trinity | Automated Resume Analysis using NLP & Machine Learning</div>', unsafe_allow_html=True)

    # Sidebar
    st.sidebar.image("https://img.icons8.com/isometric-folders/100/resume.png", width=70)
    st.sidebar.title("Configuration & Navigation")
    
    nav_option = st.sidebar.radio(
        "Select Page",
        ["Resume Skill Extractor", "Model Comparison Plan", "NLP Pipeline & Architecture"]
    )

    if nav_option == "Resume Skill Extractor":
        st.subheader("📄 Resume Analysis & Skill Extraction")
        
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("##### Step 1: Input Resume")
            input_type = st.radio("Choose Input Method", ["Upload Resume File (.pdf / .txt)", "Paste Text Directly"])
            
            resume_text = ""
            if input_type == "Upload Resume File (.pdf / .txt)":
                uploaded_file = st.file_uploader("Upload Resume File", type=["pdf", "txt"])
                if uploaded_file is not None:
                    if uploaded_file.name.endswith(".pdf"):
                        resume_text = extract_text_from_pdf(uploaded_file)
                    else:
                        resume_text = str(uploaded_file.read(), "utf-8", errors="ignore")
                    st.success(f"File '{uploaded_file.name}' loaded successfully ({len(resume_text)} characters).")
            else:
                resume_text = st.text_area("Paste Resume Text Here", height=280, placeholder="Experienced Data Scientist with 5 years in Python, Machine Learning, TensorFlow, SQL...")

            st.markdown("##### Step 2: Select Model for Classification")
            selected_model_name = st.selectbox(
                "Choose Trained NLP/ML Model",
                list(models_dict.keys()) if models_dict else ["Logistic Regression"]
            )

        with col2:
            st.markdown("##### Step 3: Analysis Results")
            analyze_button = st.button("🚀 Analyze Resume & Extract Skills", type="primary", use_container_width=True)

        if analyze_button or (resume_text and len(resume_text.strip()) > 30):
            if not resume_text or not resume_text.strip():
                st.warning("Please provide resume text or upload a valid file.")
                return

            with st.spinner("Analyzing resume text..."):
                # Extract Skills
                skills_res = extractor.extract_skills(resume_text)

                # Classify Domain
                pred_category = "Unknown"
                top_3_df = None
                
                if tfidf is not None and label_encoder is not None and selected_model_name in models_dict:
                    cleaned_t = clean_text_simple(resume_text)
                    vec = tfidf.transform([cleaned_t])
                    model = models_dict[selected_model_name]
                    
                    pred_idx = model.predict(vec)[0]
                    pred_category = label_encoder.inverse_transform([pred_idx])[0]

                    if hasattr(model, "predict_proba"):
                        probs = model.predict_proba(vec)[0]
                        top_3_idx = np.argsort(probs)[::-1][:3]
                        top_3_cats = label_encoder.inverse_transform(top_3_idx)
                        top_3_probs = probs[top_3_idx] * 100
                        top_3_df = pd.DataFrame({
                            "Job Category Domain": top_3_cats,
                            "Confidence Score (%)": [round(p, 2) for p in top_3_probs]
                        })

            # Display Extraction Results
            st.markdown("---")
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.markdown(f'<div class="metric-card"><div class="metric-value">{skills_res["total_count"]}</div><div class="metric-label">Skills Detected</div></div>', unsafe_allow_html=True)
            with m2:
                st.markdown(f'<div class="metric-card"><div class="metric-value">{skills_res["categories_found"]}</div><div class="metric-label">Skill Categories</div></div>', unsafe_allow_html=True)
            with m3:
                st.markdown(f'<div class="metric-card"><div class="metric-value">{pred_category}</div><div class="metric-label">Predicted Domain</div></div>', unsafe_allow_html=True)
            with m4:
                st.markdown(f'<div class="metric-card"><div class="metric-value">{selected_model_name}</div><div class="metric-label">Active Model</div></div>', unsafe_allow_html=True)

            st.markdown("### 🛠 Extracted Technical & Professional Skills")
            if skills_res["extracted_skills"]:
                for cat, sk_list in skills_res["extracted_skills"].items():
                    st.markdown(f'<div class="category-title">{cat} ({len(sk_list)})</div>', unsafe_allow_html=True)
                    pills_html = "".join([f'<span class="skill-badge">{s}</span>' for s in sk_list])
                    st.markdown(pills_html, unsafe_allow_html=True)
            else:
                st.info("No specific skills from predefined taxonomy detected. Try pasting a detailed resume.")

            if top_3_df is not None:
                st.markdown("### 🎯 Predicted Candidate Job Category Domain")
                c1, c2 = st.columns([1.5, 1])
                with c1:
                    st.dataframe(top_3_df, use_container_width=True, hide_index=True)
                with c2:
                    st.bar_chart(top_3_df.set_index("Job Category Domain"))

            # Export analysis report
            report_data = {
                "predicted_category": pred_category,
                "model_used": selected_model_name,
                "skills_extracted": skills_res["extracted_skills"],
                "total_skills_count": skills_res["total_count"]
            }
            st.download_button(
                "📥 Download Analysis Report (JSON)",
                data=json.dumps(report_data, indent=4),
                file_name="resume_analysis_report.json",
                mime="application/json"
            )

    elif nav_option == "Model Comparison Plan":
        st.subheader("📊 Model Comparison Plan & Performance Evaluation")
        st.markdown("Evaluation metrics (**Accuracy, Precision, Recall, F1-Score**) across all 6 ML & DL models trained on 2,484 resumes (Section 4).")

        if comp_data is not None:
            metrics_df = pd.DataFrame(comp_data["metrics"]).T
            best_m = comp_data.get("best_model", "SVM")

            st.success(f"🏆 **Best Performing Model**: **{best_m}** based on Macro/Weighted F1 Score.")

            col1, col2 = st.columns([1.2, 1])
            with col1:
                st.markdown("##### Model Metrics Summary Table")
                st.dataframe(metrics_df.style.highlight_max(axis=0, color="#d4edda"), use_container_width=True)

            with col2:
                st.markdown("##### F1-Score Comparison")
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.barplot(x=metrics_df.index, y=metrics_df['F1-Score'], palette="Blues_r", ax=ax)
                ax.set_ylabel("F1 Score")
                ax.set_ylim(0, 1.05)
                plt.xticks(rotation=30)
                st.pyplot(fig)

            st.markdown("---")
            st.markdown("##### Confusion Matrix Visualizer")
            selected_cm_model = st.selectbox("Select Model to View Confusion Matrix", list(metrics_df.index))
            
            if selected_cm_model in comp_data["confusion_matrices"]:
                cm = np.array(comp_data["confusion_matrices"][selected_cm_model])
                fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=False, fmt='d', cmap="Blues", ax=ax_cm)
                ax_cm.set_title(f"Confusion Matrix - {selected_cm_model}")
                ax_cm.set_xlabel("Predicted Label Index")
                ax_cm.set_ylabel("True Label Index")
                st.pyplot(fig_cm)
        else:
            st.warning("Model comparison data not found. Please execute model training script (`scripts/02_train_models.py`).")

    elif nav_option == "NLP Pipeline & Architecture":
        st.subheader("🏗 System Architecture & Group Workload")
        
        st.markdown("""
        ### Group Information (Group 27 NLP Trinity)
        - **Randew Rajapaksha** (CIT-24-01-0090): Logistic Regression & LSTM Models, Streamlit App Development.
        - **D.M.J.B. Disanayake** (CIT-24-01-0004): SVM & GRU Models, Data Preprocessing & Presentation.
        - **M.P. Amangi Madushani** (CIT-24-01-0361): Random Forest & BERT Models, Report & Evaluation.
        
        ---
        ### 🔄 End-to-End NLP & ML Workflow (Section 5 Diagram)
        ```
        [ Upload Resume (.pdf / .txt) ]
                      │
                      ▼
        [ Extract Raw Resume Text ]
                      │
                      ▼
        [ Preprocess Text (Cleaning, Tokenization, Stopword Removal, Lemmatization) ]
                      │
                      ▼
        [ Feature Extraction (TF-IDF Vectorization & Word Embeddings) ]
                      │
                      ▼
        [ Best Trained NLP/ML Model Inference ]
                      │
                      ├──────────────────────────┐
                      ▼                          ▼
        [ Extract & Categorize Skills ]   [ Predict Job Domain Category ]
                      │                          │
                      └──────────────────────────┘
                      │
                      ▼
        [ Display Results in Interactive Streamlit Dashboard ]
        ```
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

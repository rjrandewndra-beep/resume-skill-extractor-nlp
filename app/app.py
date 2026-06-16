import streamlit as st
import pandas as pd
import pickle
import os

st.set_page_config(
    page_title="Resume Skill Extractor",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Resume Skill Extractor NLP")
st.write("Extract professional skills and classify resume categories using Machine Learning and NLP models.")

# Sidebar controls
st.sidebar.header("Configuration")
model_option = st.sidebar.selectbox(
    "Choose Classification Model",
    ("Logistic Regression", "SVM", "BERT (Transformers)")
)

# Main UI Layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Input Resume Text")
    resume_text = st.text_area(
        "Paste the raw text of the resume here:",
        height=300,
        placeholder="E.g., Senior Software Engineer with 5+ years of experience in Python, Django, PostgreSQL, and AWS..."
    )
    
    extract_btn = st.button("Extract Skills", type="primary")

with col2:
    st.subheader("Extraction Results")
    if extract_btn and resume_text.strip():
        with st.spinner("Processing text and extracting skills..."):
            # Simple rule-based logic for demo run
            skills_database = ["python", "django", "postgresql", "aws", "machine learning", "nlp", "pytorch", "sql", "react", "typescript", "tailwind css", "agile", "scrum", "jira"]
            found_skills = [skill for skill in skills_database if skill in resume_text.lower()]
            
            st.success("Analysis Complete!")
            
            st.write(f"**Selected Model:** {model_option}")
            
            st.write("### Identified Skills:")
            if found_skills:
                for skill in found_skills:
                    st.markdown(f"- `{skill.title()}`")
            else:
                st.info("No skills from dataset catalog detected. Try pasting a matching profile.")
                
            st.write("### Predicted Profile Category:")
            if "python" in resume_text.lower() or "machine learning" in resume_text.lower():
                st.info("Data Scientist / Software Engineer")
            else:
                st.info("General Profile")
    elif extract_btn:
        st.warning("Please paste some resume text first!")

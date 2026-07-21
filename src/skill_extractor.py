import os
import re
import json
from collections import defaultdict

class ResumeSkillExtractor:
    def __init__(self, taxonomy_path=None):
        if taxonomy_path is None:
            taxonomy_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'skills_db.json')
        
        self.taxonomy = {}
        if os.path.exists(taxonomy_path):
            try:
                with open(taxonomy_path, 'r', encoding='utf-8') as f:
                    self.taxonomy = json.load(f)
            except Exception as e:
                print(f"Error loading taxonomy from {taxonomy_path}: {e}")
        
        if not self.taxonomy:
            # Fallback default taxonomy
            self.taxonomy = {
                "Programming Languages": ["python", "java", "c", "c++", "c#", "javascript", "typescript", "r", "go", "sql", "html", "css", "php", "ruby"],
                "Data Science & AI": ["machine learning", "deep learning", "nlp", "tensorflow", "pytorch", "scikit-learn", "pandas", "numpy", "tableau", "power bi", "bert"],
                "Web & Frameworks": ["react", "angular", "vue", "node.js", "django", "flask", "spring boot", "bootstrap", "rest api"],
                "Cloud & DevOps": ["aws", "azure", "gcp", "docker", "kubernetes", "jenkins", "git", "linux", "ci/cd", "devops"],
                "Databases": ["postgresql", "mysql", "mongodb", "redis", "oracle", "sql server", "sqlite"],
                "Software Engineering": ["agile", "scrum", "jira", "oop", "system design", "unit testing", "git flow"],
                "Soft Skills": ["leadership", "communication", "problem solving", "teamwork", "critical thinking", "project management"]
            }

    def extract_skills(self, text):
        if not isinstance(text, str) or not text.strip():
            return {"extracted_skills": {}, "total_count": 0, "categories_found": 0, "unique_skills": []}

        # Normalize text for matching
        text_lower = text.lower()

        extracted = defaultdict(list)
        all_found = set()

        for category, skill_list in self.taxonomy.items():
            for skill in skill_list:
                # Use word boundaries where appropriate (e.g. for C, R, Go, AWS, SQL)
                pattern = r'\b' + re.escape(skill) + r'\b'
                if re.search(pattern, text_lower):
                    extracted[category].append(skill.title() if len(skill) > 3 else skill.upper())
                    all_found.add(skill.title() if len(skill) > 3 else skill.upper())

        total_count = sum(len(skills) for skills in extracted.values())
        return {
            "extracted_skills": dict(extracted),
            "total_count": total_count,
            "categories_found": len(extracted),
            "unique_skills": sorted(list(all_found))
        }

    def highlight_skills(self, text, extracted_unique):
        if not isinstance(text, str) or not extracted_unique:
            return text
        
        highlighted = text
        for skill in extracted_unique:
            pattern = re.compile(r'\b(' + re.escape(skill) + r')\b', re.IGNORECASE)
            highlighted = pattern.sub(r'**\1**', highlighted)
        return highlighted

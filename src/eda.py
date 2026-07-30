import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_CSV = os.path.join(BASE_DIR, "dataset", "processed", "resume_dataset.csv")

def analyze_dataset():
    """
    Performs Exploratory Data Analysis (EDA) on the resume dataset.
    Calculates category distributions and text length statistics.
    """
    if not os.path.exists(PROCESSED_CSV):
        return {"error": f"Processed CSV dataset not found at {PROCESSED_CSV}."}
    
    df = pd.read_csv(PROCESSED_CSV)
    
    # Calculate category counts
    category_counts = df["Category"].value_counts().to_dict()
    
    # Calculate word and character lengths
    df["Char_Count"] = df["Resume"].fillna("").apply(len)
    df["Word_Count"] = df["Resume"].fillna("").apply(lambda x: len(x.split()))
    
    stats = {
        "total_resumes": len(df),
        "categories": list(category_counts.keys()),
        "category_distribution": category_counts,
        "word_count_stats": {
            "mean": round(float(df["Word_Count"].mean()), 2),
            "min": int(df["Word_Count"].min()),
            "max": int(df["Word_Count"].max()),
            "std": round(float(df["Word_Count"].std()), 2)
        },
        "char_count_stats": {
            "mean": round(float(df["Char_Count"].mean()), 2),
            "min": int(df["Char_Count"].min()),
            "max": int(df["Char_Count"].max())
        }
    }
    return stats

if __name__ == "__main__":
    import pprint
    print("--- Running Exploratory Data Analysis ---")
    stats = analyze_dataset()
    pprint.pprint(stats)

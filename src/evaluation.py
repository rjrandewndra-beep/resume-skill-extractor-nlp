from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

def compute_metrics(y_true, y_pred, labels=None):
    """
    Computes Accuracy, Precision, Recall, and F1 Score (overall weighted and per-class).
    Also generates the confusion matrix.
    """
    accuracy = accuracy_score(y_true, y_pred)
    
    # Overall weighted metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
    # Per-class metrics
    class_precision, class_recall, class_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "per_class": {
            "labels": labels,
            "precision": class_precision.tolist(),
            "recall": class_recall.tolist(),
            "f1_score": class_f1.tolist()
        },
        "confusion_matrix": cm.tolist()
    }
    return metrics

def display_confusion_matrix_text(metrics):
    """
    Formats and displays the confusion matrix in a readable text-based layout.
    """
    cm = metrics["confusion_matrix"]
    labels = metrics["per_class"]["labels"]
    
    print("\n--- Confusion Matrix ---")
    header = f"{'Actual / Pred':<25}" + "".join([f"{l[:10]:>12}" for l in labels])
    print(header)
    print("-" * len(header))
    for i, actual_label in enumerate(labels):
        row_str = f"{actual_label[:20]:<25}" + "".join([f"{val:>12}" for val in cm[i]])
        print(row_str)
    print("-" * len(header))

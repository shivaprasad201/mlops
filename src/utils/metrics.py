from typing import Dict
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support

def classification_metrics(y_true, y_pred) -> Dict[str, float]:
    acc = float(accuracy_score(y_true, y_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", pos_label=1)
    return {
        "accuracy": acc,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }

def confusion(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)

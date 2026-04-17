import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, accuracy_score, balanced_accuracy_score, 
    recall_score, precision_score, f1_score, roc_curve, 
    auc, roc_auc_score, average_precision_score, 
    matthews_corrcoef, brier_score_loss
)

def choose_threshold_by_recall(y_true, y_prob, target_recall=0.90, min_specificity=0.30):
    """
    Finds the optimal threshold to meet a clinical recall target.
    """
    thresholds = np.unique(np.clip(y_prob, 0.0, 1.0))
    thresholds = np.concatenate(([0.0], thresholds, [1.0]))
    
    primary_choice = None
    fallback_choice = None

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(np.int32)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * recall * prec / (recall + prec) if (recall + prec) > 0 else 0.0

        if recall >= target_recall and spec >= min_specificity:
            score = (f1, prec, spec)
            if primary_choice is None or score > primary_choice[0]:
                primary_choice = (score, float(threshold))

        youden_j = recall + spec - 1.0
        fallback_score = (youden_j, f1, -abs(float(threshold) - 0.5))
        if fallback_choice is None or fallback_score > fallback_choice[0]:
            fallback_choice = (fallback_score, float(threshold))

    return primary_choice[1] if primary_choice is not None else fallback_choice[1]

def compute_binary_metrics(y_true, y_prob, threshold):
    """
    Calculates a comprehensive suite of clinical and statistical metrics.
    """
    y_pred = (y_prob >= threshold).astype(np.int32)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "Threshold": float(threshold),
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Sensitivity_Recall": float(recall_score(y_true, y_pred)),
        "Specificity": float(tn / (tn + fp) if (tn + fp) > 0 else 0.0),
        "Precision_PPV": float(precision_score(y_true, y_pred)),
        "F1": float(f1_score(y_true, y_pred)),
        "ROC_AUC": float(roc_auc_score(y_true, y_prob)),
        "PR_AUC": float(average_precision_score(y_true, y_prob)),
        "FN": int(fn),
        "FP": int(fp)
    }
    return metrics, cm

import numpy as np
from sklearn.metrics import f1_score, accuracy_score

def multilabel_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "accuracy_samples": float(accuracy_score(y_true, y_pred))
    }

def topk_accuracy(y_true, y_prob, k=1):
    # y_true: binary matrix
    # y_prob: prob matrix
    idx = np.argsort(-y_prob, axis=1)[:, :k]
    correct = 0
    for i in range(y_true.shape[0]):
        true_labels = set(np.where(y_true[i] == 1)[0])
        pred_set = set(idx[i])
        if true_labels & pred_set:
            correct += 1
    return float(correct / y_true.shape[0])

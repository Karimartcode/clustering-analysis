import numpy as np


def silhouette_score_manual(X: np.ndarray, labels: np.ndarray) -> float:
    """Compute mean silhouette coefficient."""
    n = len(X)
    scores = []
    unique_labels = np.unique(labels)
    for i in range(n):
        same = X[labels == labels[i]]
        a = np.mean(np.linalg.norm(same - X[i], axis=1)) if len(same) > 1 else 0
        b_vals = []
        for lbl in unique_labels:
            if lbl == labels[i]:
                continue
            other = X[labels == lbl]
            b_vals.append(np.mean(np.linalg.norm(other - X[i], axis=1)))
        b = min(b_vals) if b_vals else 0
        scores.append((b - a) / max(a, b) if max(a, b) > 0 else 0)
    return float(np.mean(scores))

import numpy as np
from sklearn.datasets import make_blobs


def generate_data(n_samples=300, n_centers=4, seed=42):
    X, y = make_blobs(n_samples=n_samples, centers=n_centers, random_state=seed, cluster_std=1.0)
    return X, y

import numpy as np
from sklearn.datasets import make_blobs


def generate_data(n_samples=300, n_centers=4, seed=42):
    X, y = make_blobs(n_samples=n_samples, centers=n_centers, random_state=seed, cluster_std=1.0)
    return X, y


from sklearn.preprocessing import StandardScaler

def standardize(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X), scaler


from sklearn.cluster import KMeans

def run_kmeans(X, k):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X)
    return labels, model


def elbow_method(X, k_range=range(1, 11)):
    inertias = []
    for k in k_range:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        model.fit(X)
        inertias.append(model.inertia_)
    return list(k_range), inertias


from sklearn.metrics import silhouette_score

def silhouette_analysis(X, k_range=range(2, 11)):
    scores = []
    for k in k_range:
        labels, _ = run_kmeans(X, k)
        score = silhouette_score(X, labels)
        scores.append(score)
    return list(k_range), scores


def find_optimal_k(k_range, scores):
    return k_range[np.argmax(scores)]


from sklearn.cluster import DBSCAN

def run_dbscan(X, eps=0.5, min_samples=5):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    return labels, n_clusters, n_noise

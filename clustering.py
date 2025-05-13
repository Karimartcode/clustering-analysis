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


def cluster_stats(X, labels):
    stats = {}
    for label in set(labels):
        if label == -1:
            continue
        mask = labels == label
        cluster_data = X[mask]
        stats[label] = {
            "size": int(np.sum(mask)),
            "center": cluster_data.mean(axis=0).tolist(),
            "std": cluster_data.std(axis=0).tolist()
        }
    return stats


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_clusters(X, labels, title="Clusters", filename="clusters.png"):
    plt.figure(figsize=(8, 6))
    unique_labels = set(labels)
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
    for label, color in zip(sorted(unique_labels), colors):
        mask = labels == label
        name = f"Cluster {label}" if label >= 0 else "Noise"
        plt.scatter(X[mask, 0], X[mask, 1], c=[color], s=20, label=name, alpha=0.7)
    plt.title(title)
    plt.legend()
    plt.savefig(filename)
    plt.close()


def plot_elbow(k_range, inertias, filename="elbow.png"):
    plt.figure(figsize=(8, 4))
    plt.plot(k_range, inertias, 'bo-')
    plt.xlabel("k")
    plt.ylabel("Inertia")
    plt.title("Elbow Method")
    plt.savefig(filename)
    plt.close()


def plot_silhouette(k_range, scores, filename="silhouette.png"):
    plt.figure(figsize=(8, 4))
    plt.plot(k_range, scores, 'go-')
    plt.xlabel("k")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Analysis")
    plt.savefig(filename)
    plt.close()


def main():
    X, y_true = generate_data()
    X_scaled, _ = standardize(X)
    k_range, inertias = elbow_method(X_scaled)
    plot_elbow(k_range, inertias)
    k_range_sil, scores = silhouette_analysis(X_scaled)
    plot_silhouette(k_range_sil, scores)
    optimal_k = find_optimal_k(k_range_sil, scores)
    print(f"Optimal k: {optimal_k}")
    km_labels, km_model = run_kmeans(X_scaled, optimal_k)
    plot_clusters(X_scaled, km_labels, "K-Means", "kmeans.png")
    db_labels, n_clusters, n_noise = run_dbscan(X_scaled, eps=0.5)
    print(f"DBSCAN: {n_clusters} clusters, {n_noise} noise points")
    plot_clusters(X_scaled, db_labels, "DBSCAN", "dbscan.png")
    stats = cluster_stats(X_scaled, km_labels)
    for label, s in stats.items():
        print(f"  Cluster {label}: size={s['size']}")


if __name__ == "__main__":
    main()

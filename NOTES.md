# Clustering Notes

## Algorithm selection
- **K-Means**: fast, works well for spherical clusters
- **DBSCAN**: handles noise and arbitrary shapes
- **Agglomerative**: when hierarchy matters

## Choosing k (K-Means)
- Elbow method: plot inertia vs k, pick the 'elbow'
- Silhouette score: maximize average silhouette
- Rule of thumb: k ≈ sqrt(n/2)

## Preprocessing
- Always scale features before clustering
- Use PCA for dimensionality > 10

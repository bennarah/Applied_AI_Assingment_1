from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import numpy as np

# Load data
data = load_iris()
X = data.data
true_labels = data.target

# Apply KMeans with K=4
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X)

cluster_labels = kmeans.labels_
centroids = kmeans.cluster_centers_

print("\n--- KMEANS RESULTS ---")

# RMSE per cluster
for i in range(4):
    cluster_points = X[cluster_labels == i]

    if len(cluster_points) == 0:
        continue

    rmse = np.sqrt(np.mean((cluster_points - centroids[i]) ** 2))
    print(f"Cluster {i} RMSE:", rmse)

# Compare with true labels
print("\n--- CLUSTER vs TRUE LABELS ---")

for i in range(4):
    indices = np.where(cluster_labels == i)
    labels_in_cluster = true_labels[indices]

    print(f"\nCluster {i}:")
    print("Count:", len(labels_in_cluster))

    unique, counts = np.unique(labels_in_cluster, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"Class {u}: {c}")
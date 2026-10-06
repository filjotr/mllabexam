import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

print("Libraries imported smoothly.")

digits = load_digits()
X = digits.data
y = digits.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"Dataset successfully loaded: {X.shape[0]} samples with {X.shape[1]} dimensional pixel configurations.")
print(f"Underlying true ground-truth classes: {len(np.unique(y))}")

cluster_range = range(2, 21)
inertia_scores = []
silhouette_scores = []

print(f"{'K (Clusters)':<12} | {'Inertia':<12} | {'Silhouette Score':<16}")
print("-" * 48)

for k in cluster_range:
    kmeans = KMeans(n_clusters=k, init='k-means++',
                    random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)

    inertia = kmeans.inertia_
    silhouette = silhouette_score(X_scaled, cluster_labels)

    inertia_scores.append(inertia)
    silhouette_scores.append(silhouette)

    print(f"{k:<12} | {inertia:<12.2f} | {silhouette:<16.4f}")

plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.plot(cluster_range, inertia_scores, marker='o', color='blue', linestyle='-')
plt.title('Elbow Analysis: Inertia Reduction Plot')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia (WCSS)')
plt.xticks(cluster_range)
plt.grid(True, linestyle=':', alpha=0.6)

plt.subplot(1, 2, 2)
plt.plot(cluster_range, silhouette_scores, marker='s', color='red', linestyle='-')
plt.title('Silhouette Profile Across Cluster Quantities')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Coefficient Score')
plt.xticks(cluster_range)
plt.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.show()

optimal_k = 10
kmeans_optimal = KMeans(n_clusters=optimal_k, init='k-means++',
                        random_state=42, n_init=10)
optimal_labels = kmeans_optimal.fit_predict(X_scaled)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(9, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1],
                      c=optimal_labels, cmap='tab10',
                      alpha=0.6, s=15)
plt.colorbar(scatter, label='Mathematical Cluster Boundary ID')
plt.title(f'2D Projected PCA Spatial Cluster Map (For K={optimal_k})')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True, linestyle=':', alpha=0.5)
plt.show()

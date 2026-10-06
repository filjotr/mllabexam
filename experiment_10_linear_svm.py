import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target

y_binary = np.where(y == 0, 1, 0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"Feature matrix shape: {X_scaled.shape}")
print(f"Class distribution: Setosa ({sum(y_binary==1)}), Non-Setosa ({sum(y_binary==0)})")

svm_model = SVC(kernel='linear', C=10.0, random_state=42)
svm_model.fit(X_scaled, y_binary)

print("Linear SVM model trained successfully!")
print(f"Number of support vectors: {len(svm_model.support_vectors_)}")

plt.figure(figsize=(10, 6))
plt.scatter(X_scaled[y_binary == 1, 0], X_scaled[y_binary == 1, 1],
            color='blue', label='Setosa', edgecolors='k', s=50)
plt.scatter(X_scaled[y_binary == 0, 0], X_scaled[y_binary == 0, 1],
            color='red', label='Non-Setosa', edgecolors='k', s=50)

ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T

Z = svm_model.decision_function(xy).reshape(XX.shape)

ax.contour(XX, YY, Z, colors='black', levels=[-1, 0, 1],
           alpha=0.9, linestyles=['--', '-', '--'],
           linewidths=[1.5, 2.5, 1.5])

ax.scatter(svm_model.support_vectors_[:, 0],
           svm_model.support_vectors_[:, 1], s=150,
           linewidth=1.5, facecolors='none', edgecolors='green',
           label='Support Vectors')

plt.title("Linear SVM Decision Boundary & Margin (Setosa vs Non-Setosa)", fontsize=14)
plt.xlabel("Sepal Length (Standardized)", fontsize=12)
plt.ylabel("Sepal Width (Standardized)", fontsize=12)
plt.legend(loc='upper right')
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()

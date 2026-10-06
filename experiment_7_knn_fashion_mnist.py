import numpy as np
import matplotlib.pyplot as plt
import time
from tensorflow.keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns

(X_train_full, y_train_full), (X_test_full, y_test_full) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print(f"Full Training set shape: {X_train_full.shape}")
print(f"Full Test set shape: {X_test_full.shape}")

_, X_train_sub, _, y_train_sub = train_test_split(
    X_train_full, y_train_full, test_size=10000,
    stratify=y_train_full, random_state=42)

_, X_test_sub, _, y_test_sub = train_test_split(
    X_test_full, y_test_full, test_size=2000,
    stratify=y_test_full, random_state=42)

X_train_flat = X_train_sub.reshape(X_train_sub.shape[0], -1)
X_test_flat = X_test_sub.reshape(X_test_sub.shape[0], -1)

X_train_norm = X_train_flat / 255.0
X_test_norm = X_test_flat / 255.0

print(f"Subsampled & Processed Training Shape: {X_train_norm.shape}")
print(f"Subsampled & Processed Testing Shape: {X_test_norm.shape}")

k_values = [1, 3, 5, 7, 9, 11, 15]
accuracies = []
inference_times = []

print("Starting KNN Experiments...\n")
for k in k_values:
    print(f"Evaluating K = {k}...")
    knn = KNeighborsClassifier(n_neighbors=k, metric='minkowski', p=2, n_jobs=-1)
    knn.fit(X_train_norm, y_train_sub)
    start_time = time.time()
    y_pred = knn.predict(X_test_norm)
    end_time = time.time()
    elapsed_time = end_time - start_time
    acc = accuracy_score(y_test_sub, y_pred)
    accuracies.append(acc)
    inference_times.append(elapsed_time)
    print(f"-> Accuracy: {acc:.4f} | Inference Time: {elapsed_time:.2f} seconds\n")

fig, ax1 = plt.subplots(figsize=(10, 5))
color = 'tab:blue'
ax1.set_xlabel('Value of K')
ax1.set_ylabel('Accuracy', color=color)
ax1.plot(k_values, accuracies, marker='o', color=color, linewidth=2, label='Accuracy')
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, linestyle='--', alpha=0.6)

ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Inference Time (seconds)', color=color)
ax2.plot(k_values, inference_times, marker='s', color=color,
         linestyle='--', linewidth=2, label='Inference Time')
ax2.tick_params(axis='y', labelcolor=color)
plt.title('KNN Performance Analysis: Accuracy vs Inference Time')
fig.tight_layout()
plt.show()

best_k_idx = np.argmax(accuracies)
best_k = k_values[best_k_idx]
print(f"Optimal configuration found at K = {best_k}")

best_knn = KNeighborsClassifier(n_neighbors=best_k, n_jobs=-1)
best_knn.fit(X_train_norm, y_train_sub)
y_pred_best = best_knn.predict(X_test_norm)

print("\n### Classification Report ###")
print(classification_report(y_test_sub, y_pred_best, target_names=class_names))

cm = confusion_matrix(y_test_sub, y_pred_best)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title(f'Confusion Matrix for Best KNN (K={best_k})')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

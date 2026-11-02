# ==========================================
# 1. IMPORTS
# ==========================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ==========================================
# 2. LOAD AND INSPECT THE DATASET
# ==========================================

data = load_breast_cancer()

print(data.keys())

print("Feature data shape:", data.data.shape)
print("Target shape:", data.target.shape)

print("Target names:", data.target_names)
print("Feature names:", data.feature_names[:10])

print("First 10 target values:", data.target[:10])
print("Class counts:", np.bincount(data.target))

X = data.data
y = data.target

print("X shape:", X.shape)
print("y shape:", y.shape)

# ==========================================
# 3. TRAIN-TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training features:", X_train.shape)
print("Testing features:", X_test.shape)
print("Training labels:", y_train.shape)
print("Testing labels:", y_test.shape)

# ==========================================
# 4. BASELINE MODEL — CROSS-VALIDATION
# ==========================================

model = LogisticRegression(max_iter=5000)

cv_scores = cross_val_score(
    model,
    X_train,
    y_train,
    cv=5,
    scoring="accuracy"
)

print("\nCross-validation scores:", cv_scores)
print(f"Mean CV accuracy: {cv_scores.mean():.2%}")

# ==========================================
# 5. SCALED MODEL — CROSS-VALIDATION
# ==========================================

scaled_model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=5000)
)

scaled_cv_scores = cross_val_score(
    scaled_model,
    X_train,
    y_train,
    cv=5,
    scoring="accuracy"
)

print("\nScaled CV scores:", scaled_cv_scores)
print(f"Mean scaled CV accuracy: {scaled_cv_scores.mean():.2%}")

# ==========================================
# 6. FINAL MODEL — TEST SET EVALUATION
# ==========================================

scaled_model.fit(X_train, y_train)

y_pred_scaled = scaled_model.predict(X_test)

# ==========================================
# 7. FINAL RESULTS
# ==========================================

test_accuracy = accuracy_score(y_test, y_pred_scaled)

print(f"\nScaled test accuracy: {test_accuracy:.2%}")

cm = confusion_matrix(y_test, y_pred_scaled)

print("\nConfusion matrix:")
print(cm)
print("Class order:", data.target_names)

plt.figure(figsize=(6, 5))

plt.imshow(cm, interpolation="nearest" , cmap = "Blues")
plt.title("Confusion Matrix — Logistic Regression")
plt.colorbar()

classes = data.target_names
plt.xticks([0, 1], classes)
plt.yticks([0, 1], classes)

plt.xlabel("Predicted label")
plt.ylabel("Actual label")

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        color = "white" if cm[i, j] > cm.max() / 2 else "black"
        plt.text(j, i, cm[i, j], ha="center", va="center" , color=color)

plt.tight_layout()

plt.savefig("confusion_matrix.png", dpi=150)

plt.show()

print("\nClassification report:")
print(classification_report(
    y_test,
    y_pred_scaled,
    target_names=data.target_names
))
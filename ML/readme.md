# ==============================
# 1. Data Analysis
# ==============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
link = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
df = pd.read_csv(link, header="infer", delimiter=";")

# Dataset summary
print("\n========= Dataset summary ========= \n")
df.info()
print("\n========= A few first samples ========= \n")
print(df.head())

# Form input and output arrays
X = df.drop("quality", axis=1)
Y = df["quality"]

print("\n========= Wine Qualities ========= \n")
print(Y.value_counts())

# Convert to binary classification: 0 = bad (<=5), 1 = good (>5)
Y = np.array([0 if val <= 5 else 1 for val in Y])
print("\n========= Samples per class ========= \n")
print(pd.Series(Y).value_counts())

# Statistical analysis
plt.figure(figsize=(12,6))
sns.boxplot(data=X, orient="v", palette="Set1", width=1.5, notch=True)
plt.xticks(rotation=90)
plt.title("Boxplot of Input Features")
plt.show()

plt.figure(figsize=(10,8))
corr = X.corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix of Features")
plt.show()

# ==============================
# 2. Classification
# ==============================
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# 2.1 Data split
# First split into training+validation (2/3) and test (1/3)
Xa, Xt, Ya, Yt = train_test_split(X, Y, test_size=1/3, shuffle=True, stratify=Y, random_state=42)
# Split training into training and validation (1/2 of training)
Xa, Xv, Ya, Yv = train_test_split(Xa, Ya, test_size=0.5, shuffle=True, stratify=Ya, random_state=42)

print("\n========= Sizes =========")
print(f"Training: {Xa.shape}, Validation: {Xv.shape}, Test: {Xt.shape}")

# 2.2 k-NN Classification
# Starter k = 3
k = 3
clf = KNeighborsClassifier(n_neighbors=k)
clf.fit(Xa, Ya)
Ypred_v = clf.predict(Xv)
error_v = 1 - accuracy_score(Yv, Ypred_v)
print(f"\nValidation error for k={k}: {error_v:.4f}")

# Evaluate for multiple k
k_vector = np.arange(1, 40, 2)
error_train = []
error_val = []

for k in k_vector:
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(Xa, Ya)
    Ypred_train = clf.predict(Xa)
    Ypred_val = clf.predict(Xv)
    error_train.append(1 - accuracy_score(Ya, Ypred_train))
    error_val.append(1 - accuracy_score(Yv, Ypred_val))

# Plot training and validation error
plt.figure(figsize=(10,6))
plt.plot(k_vector, error_train, label="Training Error", marker='o')
plt.plot(k_vector, error_val, label="Validation Error", marker='o')
plt.xlabel("k")
plt.ylabel("Error Rate")
plt.title("k-NN: Training vs Validation Error")
plt.legend()
plt.show()

# Select best k
err_min = np.min(error_val)
ind_opt = np.argmin(error_val)
k_star = k_vector[ind_opt]
print(f"Optimal k (based on validation error): {k_star}, with error: {err_min:.4f}")

# Test error with optimal k
clf_opt = KNeighborsClassifier(n_neighbors=k_star)
clf_opt.fit(Xa, Ya)
Ypred_test = clf_opt.predict(Xt)
error_test = 1 - accuracy_score(Yt, Ypred_test)
print(f"Test error with k={k_star}: {error_test:.4f}")

# 2.3 Normalize the data
sc = StandardScaler(with_mean=True, with_std=True)
sc.fit(Xa)
Xa_n = sc.transform(Xa)
Xv_n = sc.transform(Xv)
Xt_n = sc.transform(Xt)

# Repeat k-NN on normalized data
error_train_n = []
error_val_n = []

for k in k_vector:
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(Xa_n, Ya)
    Ypred_train_n = clf.predict(Xa_n)
    Ypred_val_n = clf.predict(Xv_n)
    error_train_n.append(1 - accuracy_score(Ya, Ypred_train_n))
    error_val_n.append(1 - accuracy_score(Yv, Ypred_val_n))

# Plot normalized errors
plt.figure(figsize=(10,6))
plt.plot(k_vector, error_train_n, label="Training Error (Normalized)", marker='o')
plt.plot(k_vector, error_val_n, label="Validation Error (Normalized)", marker='o')
plt.xlabel("k")
plt.ylabel("Error Rate")
plt.title("k-NN on Normalized Data: Training vs Validation Error")
plt.legend()
plt.show()

# Test error with normalization and optimal k
clf_opt_n = KNeighborsClassifier(n_neighbors=k_star)
clf_opt_n.fit(Xa_n, Ya)
Ypred_test_n = clf_opt_n.predict(Xt_n)
error_test_n = 1 - accuracy_score(Yt, Ypred_test_n)
print(f"Test error (normalized data) with k={k_star}: {error_test_n:.4f}")


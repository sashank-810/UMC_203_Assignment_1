# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---


# Question 1
import oracle 
data = oracle.q1_fish_train_test_data(23634)

import numpy as np

training_img = np.array(data[1])
training_label = np.array(data[2])
testing_img = np.array(data[3])
testing_label = np.array(data[4])
training_img = training_img.reshape(training_img.shape[0], -1)
testing_img = testing_img.reshape(testing_img.shape[0], -1)
print(training_img.shape)


# +
def compute_meanandcovariance(X, y, num_samples):
    class_means = []
    class_covs = []
    for i in np.unique(y):
        X_class = X[y == i][:num_samples]
        class_means.append(np.mean(X_class, axis=0))
        class_covs.append(np.cov(X_class, rowvar=False))
    # print(class_covs[0].shape)
    return np.array(class_means), np.array(class_covs)
sample_size = [50,100,500,1000,2000,4000]
l2_norm = []
frob_norm = []
for n in sample_size:
    means, covs =compute_meanandcovariance(training_img, training_label, n)
    mean_norms = [np.linalg.norm(mean) for mean in means]
    cov_norms = [np.linalg.norm(cov, 'fro') for cov in covs]
    l2_norm.append(mean_norms)
    frob_norm.append(cov_norms)
l2_norm = np.array(l2_norm)
frob_norm = np.array(frob_norm) 


import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
for i in range(4) :
    plt.plot(sample_size, l2_norm[:,i], marker = 'o', label=f'Class {i}')
plt.xlabel('Number of samples')
plt.ylabel('L2 norm of mean')
plt.title('L2 norm of means vs Size of training set')
plt.legend()
plt.show()
# -

plt.figure(figsize=(12, 6))
for i in range(4) :
    plt.plot(sample_size, frob_norm[:,i], marker = 'o', label=f'Class {i}')
plt.xlabel('Number of samples')
plt.ylabel('Frobenius Norm of Covariance')
plt.title('Frobenius Norm of Covariance vs Size of training set')
plt.legend()
plt.show()


# +
def compute_scatter_matrices(X, y):
    classes = np.unique(y)
    mean_overall = np.mean(X, axis=0)
    S_W = np.zeros((X.shape[1], X.shape[1]))
    S_B = np.zeros((X.shape[1], X.shape[1]))

    for cls in classes:
        X_cls = X[y == cls]
        mean_cls = np.mean(X_cls, axis=0)
        S_W += np.cov(X_cls.T) * (X_cls.shape[0] - 1)

        mean_diff = (mean_cls - mean_overall).reshape(-1, 1)
        S_B += X_cls.shape[0] * (mean_diff @ mean_diff.T)

    return S_W, S_B

S_W, S_B = compute_scatter_matrices(training_img, training_label)


# -

def fisher_discriminant(S_W, S_B, num_classes):
    eigvals, eigvecs = np.linalg.eig(np.linalg.inv(S_W) @ S_B)


    sorted_indices = np.argsort(-eigvals)
    eigvecs = eigvecs[:, sorted_indices[:num_classes - 1]]
    return eigvecs
# S_W, S_B = compute_scatter_matrices(training_img, training_label)
# Wfull = fisher_discriminant(S_W, S_B, num_classes=4)
# print(Wfull)


def compute_objective_value(W, S_W, S_B):
    num = np.linalg.det(W.T @ S_B @ W)
    den = np.linalg.det(W.T @ S_W @ W)
    return num / den



n_values = [2500, 3500, 4000, 4500, 5000]
objective_values = {n: [] for n in n_values}
W_values = {n: [] for n in n_values} 
for n in n_values:
    print(f"Processing n = {n}...")

    for _ in range(20 if n != 5000 else 1):
        subset_indices = np.hstack([np.random.choice(np.where(training_label == cls)[0], n, replace=False) for cls in range(4)])
        X_subset, y_subset = training_img[subset_indices], training_label[subset_indices]
        S_W, S_B = compute_scatter_matrices(X_subset, y_subset)
        W = fisher_discriminant(S_W, S_B, num_classes=4)
        W_values[n].append(W)
        J_W = compute_objective_value(W, S_W, S_B)
        objective_values[n].append(J_W)


plt.figure(figsize=(10, 6))
plt.boxplot([objective_values[n] for n in n_values], labels=n_values)
plt.xlabel("n")
plt.ylabel("Objective value")
plt.title("Box Plot for different values of n")
plt.grid(True)
plt.show()

# +
best_weight_vectors = {}

for n in n_values:
    max_JW = max(objective_values[n]) 
    max_indices = [i for i, val in enumerate(objective_values[n]) if val == max_JW]
    best_Ws = np.array([W_values[n][i] for i in max_indices])
    best_weight_vectors[n] = np.mean(best_Ws, axis=0)
    print(f"Best J(W) for n={n}: {max_JW:.6f}, Averaged W Shape: {best_weight_vectors[n].shape}")


# +
projected_data = {}
for n in n_values:
    W_avg = best_weight_vectors[n]
    X_projected = training_img @ W_avg
    projected_data[n] = X_projected
    print(f"Projection shape for n={n}: {X_projected.shape}")  # Should be (20000, 3)


# +
mean_projections = {}
for n in n_values:
    X_proj = projected_data[n]
    mean_proj = np.array([np.mean(X_proj[training_label == cls], axis=0) for cls in range(4)])
    mean_projections[n] = mean_proj
    print(f"Mean projection for n={n}:\n{mean_proj}\n")


# +
thresholds = {}
for n in n_values:
    mean_proj = mean_projections[n]
    class_thresholds = {}

    for d in range(3):
        sorted_classes = np.argsort(mean_proj[:, d])
        for i in range(3):
            cls1, cls2 = sorted_classes[i], sorted_classes[i+1]
            threshold = (mean_proj[cls1, d] + mean_proj[cls2, d]) / 2
            class_thresholds[f"Discriminant {d+1}: {cls1}-{cls2}"] = threshold
    thresholds[n] = class_thresholds
for n, thresh in thresholds.items():
    print(f"\nThresholds for n={n}:")
    for key, value in thresh.items():
        print(f"{key} → {value:.6f}")


# +
from mpl_toolkits.mplot3d import Axes3D
n_vis = 5000
X_proj = projected_data[n_vis]
labels = training_label
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
colors = ['r', 'g', 'b', 'y']
for cls in range(4):
    idx = (labels == cls)
    ax.scatter(X_proj[idx, 0], X_proj[idx, 1], X_proj[idx, 2], c=colors[cls], label=f'Class {cls}', alpha=0.5)
ax.set_xlabel("FLD Dimension 1")
ax.set_ylabel("FLD Dimension 2")
ax.set_zlabel("FLD Dimension 3")
ax.set_title(f"3D Scatter Plot of Projected Points (n={n_vis})")
ax.legend()
plt.show()


# +
projected_test_data = {}
for n in n_values:
    W_avg = best_weight_vectors[n]
    X_test_projected = testing_img.reshape(testing_img.shape[0], -1) @ W_avg
    projected_test_data[n] = X_test_projected
    print(f"Projected test data shape for n={n}: {X_test_projected.shape}")


# +
def multi_class_fld(test_images, weight_matrix, projected_class_means):
    flattened_test_images = test_images.reshape(test_images.shape[0], -1)
    projected_test_images = flattened_test_images @ weight_matrix
    predicted_labels = []
    for image in projected_test_images:
        min_distance = float('inf')
        predicted_label = None
        for class_label, projected_mean in projected_class_means.items():
            distance = np.linalg.norm(image - projected_mean)
            if distance < min_distance:
                min_distance = distance
                predicted_label = class_label
        predicted_labels.append(predicted_label)
    return predicted_labels

def evaluate_FLD(test_images, test_labels, weight_matrix, projected_class_means):
    predicted_labels = multi_class_fld(test_images, weight_matrix, projected_class_means)
    correct_predictions = 0
    for predicted_label, true_label in zip(predicted_labels, test_labels):
        if predicted_label == true_label:
            correct_predictions += 1
    accuracy = correct_predictions / len(test_labels)
    return accuracy

if isinstance(mean_projections[5000], np.ndarray):
    projected_class_means = {i: mean_projections[5000][i] for i in range(len(mean_projections[5000]))}
else:
    projected_class_means = mean_projections[5000]
W = best_weight_vectors[5000]
accuracy = evaluate_FLD(testing_img, testing_label, W, projected_class_means)
print(f"Accuracy of FLD on test data: {accuracy}")





# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---


# Question 3

import oracle
data = oracle.q3_hyper(23634)
criterion, splitter, maxdepth = data[0], data[1], data[2]
print(criterion, splitter, maxdepth)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# +
file_path = '/home/bhargavsk/Desktop/Direc3/heart+disease/processed.cleveland.data'
df = pd.read_csv(file_path, header=None)
df.columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
df



# +
categorical_columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal', 'target']
integer_columns = [col for col in df.columns if col not in categorical_columns]
for col in df.columns:
    if col in categorical_columns:
        mode_value = df[col].mode()[0]
        df[col] = df[col].replace('?', mode_value)
    else:
        mean_value = df[col].astype(float).mean()
        df[col] = df[col].replace('?', mean_value)
df['ca'] = df['ca'].astype('float64')
df['thal'] = df['thal'].astype('float64')
df

# Data cleanup is Done with this cell

# +
from sklearn.model_selection import train_test_split
training, testing = train_test_split(df, test_size=0.2, random_state=711)
training['target'] = training['target'].replace([2, 3, 4], 1)
testing['target'] = testing['target'].replace([2, 3, 4], 1)
traininglabels = training['target']
testinglabels = testing['target']
training = training.drop(columns=['target'])
testing = testing.drop(columns=['target'])
training = training.apply(pd.to_numeric, errors='coerce')
traininglabels = pd.to_numeric(traininglabels, errors='coerce')
testing = testing.apply(pd.to_numeric, errors='coerce')
testinglabels = pd.to_numeric(testinglabels, errors='coerce')
training.to_csv('training.csv', index=False)
testing.to_csv('testing.csv', index=False)
# -

from sklearn import tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from dtreeviz import model
import sklearn

criterion, splitter, maxdepth = data[0], data[1], data[2]
Tree = tree.DecisionTreeClassifier(criterion=criterion, splitter=splitter, max_depth=maxdepth)
Tree = Tree.fit(training, traininglabels)
testpred = Tree.predict(testing)
testtrue = testinglabels
accuracy = accuracy_score(testtrue, testpred)
precision = precision_score(testtrue, testpred)
recall = recall_score(testtrue, testpred)
f1 = f1_score(testtrue, testpred)
print(accuracy)
print(precision)
print(recall)
print(f1)

# +
import graphviz
import dtreeviz
feature_names = training.columns.tolist()
print(feature_names)
viz_model = dtreeviz.model(Tree,
                           X_train=training, 
                           y_train=traininglabels,
                           feature_names=feature_names,
                           class_names=["0", "1"])

viz_model.view()

# -

importance = Tree.feature_importances_
indices = np.argsort(importance)[::-1]
print("Feature ranking:")
for f in range(training.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importance[indices[f]]))
    print(training.columns[indices[f]])
    print("\n")

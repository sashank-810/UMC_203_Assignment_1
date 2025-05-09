# UMC_203_Assignment_1
**Course**: UMC 203 — Artificial Intelligence and Machine Learning  
**Student**: Kolipaka Bhargav Sashank  
**SR Number**: 23634  
**Date**: March 2025  

## 📄 Repository Contents

This repository contains the solution and report for Assignment 1 of the AI/ML course at IISc. The assignment explores fundamental techniques in classification using both linear discriminant methods and decision trees.

### Files Included

- `AIML_2025_A1_23634.pdf` — Detailed report of the assignment (Questions 1 and 3 attempted).
- `AIML_2025_Assignment_1.pdf` — The original assignment questions.
- `AIML_2025_A1_23634.py` — Python code used to solve the assignment problems.

##  Questions Attempted

### Question 1: Multi-Class Fisher Linear Discriminant

- Dataset: Subset of CelebA dataset via `oracle.q1_fish_train_test_data`.
- Tasks performed:
  - Analyzed L2 norms of class mean vectors and Frobenius norms of covariance matrices for increasing sample sizes.
  - Implemented multi-class FLD by computing within-class and between-class scatter matrices.
  - Computed optimal projection matrix `W` via eigen-decomposition.
  - Chose thresholds for classification based on class means in projected space.
  - Visualized results using box plots and 3D scatter plots.
  - Evaluated accuracy on test data.
- Achieved test accuracy: **81.9%**

### Question 3: Decision Tree for Heart Disease Classification

- Dataset: UCI Heart Disease dataset.
- Tasks performed:
  - Preprocessed dataset (handled missing values, encoded categories, normalized).
  - Used `oracle.q3_hyper` to get hyperparameters (criterion, splitter, max depth).
  - Trained `DecisionTreeClassifier` using scikit-learn.
  - Visualized the trained tree using `dtreeviz`.
  - Computed performance metrics: accuracy, precision, recall, and F1-score.
  - Ranked features by importance.

| Metric     | Value        |
|------------|--------------|
| Accuracy   | 0.770        |
| Precision  | 0.773        |
| Recall     | 0.654        |
| F1-Score   | 0.708        |
| Top Feature | `cp` (chest pain type) |

## Question Not Attempted

- **Question 2**: Bayes Classification with Reject Option (not implemented in this submission).

## ⚙️ Requirements

- Python 3.12.8
- Libraries:
  - `numpy`, `pandas`, `matplotlib`, `sklearn`, `dtreeviz`, `graphviz`

## Notes

- All datasets were accessed as instructed using the provided `oracle` functions.
- The CelebA dataset was used for FLD; the UCI Heart Disease dataset was used for decision trees.
- Extensive visualizations were used to analyze model behavior and effectiveness.

## Contact

For any queries, please reach out to:  
**Kolipaka Bhargav Sashank**  
**SR Number**: 23634  
**Mail :- bhargavsk@iisc.ac.in**  
**IISc, Bangalore**

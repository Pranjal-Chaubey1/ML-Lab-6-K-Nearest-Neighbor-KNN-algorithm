# Diabetes Prediction using K-Nearest Neighbors (KNN)

A machine learning notebook (ML Lab 5) that cleans a diabetes dataset, scales the features, trains a **K-Nearest Neighbors classifier** and tunes it with cross-validation and `GridSearchCV`.

**Author:** Pranjal Chaubey
**Roll No:** 1
**Batch:** A-A1
**Notebook:** `ML_lab_5.ipynb`

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Tech Stack](#tech-stack)
- [Project Workflow](#project-workflow)
- [Results](#results)
- [Key Insights](#key-insights)
- [How to Run](#how-to-run)
- [Project Structure](#project-structure)
- [Limitations and Future Work](#limitations-and-future-work)

---

## Overview

The goal is to predict whether a patient has diabetes (`Outcome` = 1) or not (`Outcome` = 0) from 8 medical measurements. The notebook covers:

- Data inspection and cleaning of hidden missing values
- Stratified train-test split and feature scaling
- Training and evaluating a KNN classifier (k = 5)
- Finding the best `k` using a train-vs-test accuracy curve and 10-fold cross-validation
- Hyperparameter tuning with `GridSearchCV`

## Dataset

| Property | Value |
|---|---|
| File | `diabetes.csv` (Pima Indians Diabetes format) |
| Rows | 768 |
| Columns | 9 (8 features + 1 target) |
| Target | `Outcome` (0 = no diabetes, 1 = diabetes) |

**Features:** Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age.

**Class balance:** about 65% No Diabetes and 35% Diabetes.

**Hidden missing values:** the data has no NaN values, but these columns contain impossible zeros:

| Column | Invalid zeros |
|---|---|
| Glucose | 5 |
| BloodPressure | 35 |
| SkinThickness | 227 |
| Insulin | 374 |
| BMI | 11 |

> The notebook reads the file from `/content/diabetes (6) - diabetes (6).csv` (a Google Colab path). Change the path in the `pd.read_csv(...)` cell to match where your CSV is stored.

## Tech Stack

- Python 3
- pandas
- NumPy
- matplotlib
- seaborn
- scikit-learn
- Jupyter Notebook / Google Colab

## Project Workflow

1. **Load and inspect** the data (`head`, `tail`, `shape`, `info`, `describe`, `isnull`).
2. **Clean the data:** replace invalid zeros in 5 columns with `NaN`, then fill each with the column median.
3. **Prepare data:** `X` = 8 features, `y` = `Outcome`.
4. **Split:** 80/20 with `random_state=42` and `stratify=y` (614 train / 154 test).
5. **Scale:** `StandardScaler` fitted on the training data only.
6. **Train:** `KNeighborsClassifier(n_neighbors=5)`.
7. **Evaluate:** accuracy, precision, recall, F1, classification report, confusion matrix.
8. **Choose k:** train vs test accuracy for k = 1 to 30.
9. **Cross-validate:** `StratifiedKFold` with 10 splits for k = 1 to 30.
10. **Tune:** `GridSearchCV` (5-fold) over `n_neighbors`, `weights`, `metric` and `p`.

## Results

**Baseline model (k = 5), test set:**

| Metric | Score |
|---|---|
| Accuracy | 0.753 |
| Precision (Diabetes) | 0.66 |
| Recall (Diabetes) | 0.611 |
| F1-score (Diabetes) | 0.635 |

**Confusion matrix:**

| | Predicted No Diabetes | Predicted Diabetes |
|---|---|---|
| **Actual No Diabetes** | 83 | 17 |
| **Actual Diabetes** | 21 | 33 |

**Choosing k:**

| Method | Best k | Accuracy |
|---|---|---|
| Test-set accuracy curve | 8 | 0.766 (test) |
| 10-fold cross-validation | 24 | 0.777 (CV) |
| `GridSearchCV` | 19 | 0.777 (CV) |

**Best parameters from `GridSearchCV`:** `n_neighbors=19`, `weights='uniform'`, `metric='euclidean'` (the `p` value has no effect with the euclidean metric).

## Key Insights

- Zeros in medical columns are really missing data, so `isnull()` alone gives a false sense of completeness.
- Scaling matters for KNN because it is distance-based; unscaled Insulin would dominate every other feature.
- At k = 1 the model overfits (training accuracy 1.00 against about 0.72 on the test set); larger k closes the gap.
- The model finds healthy patients better (F1 0.81) than diabetic ones (F1 0.63), and 21 diabetic patients were missed.
- Tuning more parameters gave about the same cross-validated accuracy (0.777) as plain cross-validation, so the data is the main limit.

## How to Run

```bash
# 1. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn jupyter

# 2. Put the diabetes CSV next to the notebook and update the path in the read_csv cell

# 3. Launch Jupyter and run all cells
jupyter notebook ML_lab_5.ipynb
```

You can also upload the notebook and CSV to Google Colab and run it there.

## Project Structure

```
.
├── ML_lab_5.ipynb      # Full analysis and model
├── diabetes.csv        # Dataset (add this yourself)
└── README.md
```

## Limitations and Future Work

**Known issues in the current notebook**
- The final tuned model (`best_knn`) makes predictions, but its test accuracy, precision and recall are never printed.
- The "Scaled Mean" / "Scaled Std" comparison uses `.mean()` on a NumPy array, so it shows one overall value instead of per-feature values (use `axis=0`).
- `scaler.fit_transform(X_test)` re-fits the scaler on test data; it should only use `transform`.
- Medians are computed on the full dataset before the split, which lets a little test information into training.
- The best k from the accuracy curve is chosen using the test set, which makes that score slightly optimistic.

**Improvements to try**
- Evaluate the tuned model on the test set and compare it with the k = 5 baseline.
- Put imputation, scaling and KNN in a single scikit-learn `Pipeline`.
- Focus on recall for the Diabetes class, since missed patients are the costly error.
- Compare with Logistic Regression, Random Forest or Gradient Boosting.

# 🎗️ Breast Cancer Classification — SVM with Preprocessing, Feature Selection & Visualization

This project demonstrates a **complete machine learning workflow** using the **Breast Cancer dataset** and **Support Vector Machines (SVM)**.  
It covers **data exploration**, **outlier handling**, **skewness correction**, **feature selection**, **model training**, **hyperparameter tuning**, and **decision boundary visualization**.

---

## 📂 Project Overview
The project follows these main steps:
1. **Data Loading & Exploration** — Load and inspect the dataset.
2. **Data Cleaning** — Handle outliers and correct skewness in numerical features.
3. **Feature Selection** — Remove highly correlated and redundant features.
4. **Preprocessing** — Scale features and encode target labels.
5. **Model Training** — Train SVM models with different kernels.
6. **Hyperparameter Tuning** — Use Grid Search to optimize parameters.
7. **Evaluation** — Assess the model using cross-validation and test metrics.
8. **Visualization** — Plot decision boundaries with PCA for interpretability.

---

## 📊 Dataset Description
- **Samples:** ~570 breast cancer cases  
- **Features:** Tumor measurements such as radius, texture, perimeter, area, smoothness, concavity, symmetry, etc.  
- **Target Classes:**  
  - **B** — Benign  
  - **M** — Malignant  

---

## 🛠️ Workflow Steps

### 1. Data Loading & Exploration
- Load dataset from CSV.
- Drop `id` column (non-predictive).
- Check for missing values and duplicates.
- View class distribution to check for imbalance.

### 2. Data Cleaning
- **Outlier detection & capping:** Apply IQR method to cap extreme values.
- **Skewness correction:** Use log or log1p transformations for skewed features.

### 3. Feature Selection
- Compute correlation matrix to identify redundant features.
- Drop features with correlation > 0.9 or those deemed less relevant.

### 4. Preprocessing
- **Feature scaling:** Standardize features with `StandardScaler`.
- **Label encoding:** Convert `diagnosis` to numeric labels (B = 0, M = 1).

### 5. Model Training
- Train SVM models with:
  - **RBF kernel**
  - **Linear kernel**
- Evaluate using 5-fold cross-validation accuracy.

### 6. Hyperparameter Tuning
- Perform `GridSearchCV` on:
  - `C` = [0.1, 1, 10, 100]
  - `gamma` = ['scale', 0.01, 0.001, 0.0001]
  - `kernel` = ['rbf', 'linear']
- Optimize for **F1-macro** score.
- it gives best params 
  - `C` = 1
  - `gamma` = `scale`
  - `kernel` = `rbf`

### 7. Evaluation
- Confusion matrix (heatmap).
- Classification report with precision, recall, and F1-score.

### 8. Visualization
- Apply PCA to reduce features to 2 dimensions.
- Plot decision boundary for the SVM model in PCA space.

---

## 📦 Requirements
- pandas
- numpy
- seaborn
- matplotlib
- scikit-learn
- scipy
- joblib
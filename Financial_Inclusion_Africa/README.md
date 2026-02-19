# Financial Inclusion Prediction
A supervised machine learning project focused on predicting individual bank account ownership using structured survey data from a Zindi financial inclusion challenge.


## Overview

This project was completed as part of my responsibilities at NgaoLabs.

The objective was to build and evaluate machine learning models capable of predicting whether an individual owns a bank account (`bank_account`).

The dataset was sourced from a financial inclusion challenge hosted on Zindi. The task is a binary classification problem focused on financial access prediction.

---

## Problem Statement

Predict whether an individual has a bank account.

**Target Variable**

- `1` → Has bank account  
- `0` → Does not have bank account  

---

## Dataset

The dataset consists of:

- `Train.csv` – Features and target variable . Used to fit models's performance.
- `Test.csv` – Features only  . Used by the model for prediction

More information about the dataset is found in the wrangling notebook of this project (still working on it)

---
## Project Workflow

The project followed a structured and iterative machine learning pipeline:

---

### 1. Data Cleaning and Preparation

- Loaded datasets using `pandas`.
- Created a reusable `wrangle()` function for consistent data loading and cleaning.
- Dropped high-cardinality feature `uniqueid` to prevent noise and dimensional explosion.
- Verified absence of missing values across features and target.
- Checked dataset shape and data types.
- Examined cardinality of categorical variables.
- Assessed multicollinearity using correlation matrix and heatmap.
- Split data into: - Features (`X_train`) - Target (`y_train`)
- Applied stratified train-validation split (`train_test_split`). Now we have a train set to fit models and a validation set to evaluate models's performance 
---

### 2. Exploratory Data Analysis (EDA)

- Analyzed dataset structure using `.info()` and `.shape`.
- Confirmed class imbalance via count visualization (`sns.catplot`).
- Evaluated numerical feature relationships using correlation heatmap.
- Identified categorical-heavy feature space requiring encoding.

EDA informed encoding strategy and feature elimination decisions.

---

### 3. Feature Encoding and Scaling

- Implemented a reusable preprocessing function:
- Encoding is used to transform target variables to binary columns ML models can use.
- Label Encoding is carried on categorical variables with order
- One Hot ENcoding is carried on categorical variables with no order

**Target Encoding**: - Applied `LabelEncoder` to `bank_account`.

**Feature Encoding**
- One-hot encoding for multi-class categorical variables:
  - `relationship_with_head`
  - `marital_status`
  - `education_level`
  - `job_type`
  - `country`
- Label encoding for binary categorical variables:
  - `location_type`
  - `cellphone_access`
  - `gender_of_respondent`

**Scaling**
- Applied `MinMaxScaler` to normalize all numerical and encoded features.
- Ensured consistent transformation for train and test datasets.

---

### 4. Feature Selection

Applied multiple feature selection strategies:

**Filter Method**
- `SelectKBest` with `mutual_info_classif`.
- Retained features with score > 0.001.

**Embedded Methods**
- L1-regularized Logistic Regression (Lasso) to eliminate zero-coefficient features.
- Tree-based feature importance from:
  - Random Forest
  - Gradient Boosting
  - Extra Trees

Reduced feature subsets were evaluated against baseline models to measure impact.

---

### 5. Model Training and Evaluation

Each model followed a structured approach:
- Baseline model training
- Validation set evaluation
- Iterative refinement (where applicable)

**Models Implemented**

- Logistic Regression (baseline + L1 regularization)
- Random Forest (baseline + reduced feature version)
- Gradient Boosting (baseline + reduced feature version)
- K-Nearest Neighbors (multiple `k` values tested)
- Extra Trees (baseline + reduced feature version)
- XGBoost (baseline + hyperparameter tuning via `GridSearchCV`)

---

### 6. Model Comparison Using Standardized Metrics

All models were evaluated using:

- Accuracy
- F1 Score
- ROC-AUC

Due to class imbalance:
- F1 Score and ROC-AUC were prioritized over accuracy.

Performance was compared across:
- Baseline vs refined versions
- Feature-reduced vs full feature sets
- Tuned vs untuned models

## Key Findings

- Ensemble tree-based models outperformed linear models (The XgBoost model, optimized, performed better than the rest).
- Feature elimination did not consistently improve ensemble performance (Observed in the GradientBoost model where baseline model outperformed it's improved version).
- XGBoost required hyperparameter tuning for optimal results.
- Accuracy alone was not a reliable metric due to class imbalance.

---

### 7. Final Model Selection and Submission

- Selected optimized XGBoost model for final predictions.
- Generated predictions for test dataset.
- Constructed submission format:
  - `uniqueid + country`
  - Predicted `bank_account`
- Exported final submission file:

## Authors 
Dave Karanja


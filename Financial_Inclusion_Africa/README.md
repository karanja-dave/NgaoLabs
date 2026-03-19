# Financial Inclusion Prediction
A supervised machine learning project focused on predicting individual bank account ownership using structured survey data from a Zindi financial inclusion challenge

**Objective:** Predict whether an individual owns a bank account using structured survey data from Zindi’s Financial Inclusion Challenge. This project was completed as part of my work at NgaoLabs.

---

## Executive Summary

- **Problem Type:** Binary Classification (bank account ownership)  
- **Dataset:** Structured survey data (`Train.csv` with target, `Test.csv` without target)  
- **Final Model:** Optimized XGBoost  
- **Key Metric:** ROC-AUC prioritized due to class imbalance  
- **Outcome:** Model achieved [insert ROC-AUC/F1 score], providing actionable insights for identifying populations without financial access.

---

## Problem to be solved

Financial inclusion is a critical challenge, particularly in regions where large populations remain unbanked. Predicting bank account ownership enables targeted interventions and better policy or business strategies.

---

## Project Structure

| Folder/File | Description |
|--------------|-------------|
| `data/` | Contains all datasets used in the project. |
| `data/train.csv` | Training dataset containing both features and the target variable (`bank_account`). Used for model training and validation. |
| `data/test.csv` | Test dataset containing only features. Used to generate predictions for submission. |
| `documentation/` | Holds project documentation and explanatory materials. |
| `README.md` | Detailed project description including workflow, modeling approach, and results. |
| `codes.py` | Main project script implementing the full machine learning pipeline: data cleaning, exploratory data analysis, feature encoding, feature selection, model training, evaluation, and submission generation. |
| `requirements.txt` | Lists all Python dependencies required to run the project, allowing the environment to be reproduced using `pip install -r requirements.txt`. |

---
## Project Workflow

**Pipeline Overview:**

### 1. Data Cleaning & Preparation
- Created reusable `wrangle()` function  
- Dropped high-cardinality and irrelevant features  
- Verified missing values and feature types  

### 2. Exploratory Data Analysis (EDA)
- Identified class imbalance  
- Evaluated numerical and categorical feature relationships  
- Informed encoding and feature selection strategy  

### 3. Feature Encoding & Scaling
- One-hot encoding for multi-class categorical features  
- Label encoding for binary features  
- MinMax scaling for numerical and encoded features  

### 4. Feature Selection
Applied multiple feature selection strategies:

1. Filter Method
- SelectKBest with mutual_info_classif.
- Retained features with score > 0.001.
- Embedded Methods

2. L1-regularized Logistic Regression (Lasso) to eliminate zero-coefficient features.
3. Pruning was done for features used in all models except logistic regression

### 5. Modeling & Evaluation
- Separated the target from the feature variables
- Split the train data into validation and train sets 
- Fitted Models: Logistic Regression, Random Forest, Gradient Boosting, Extra Trees, KNN, XGBoost  
- Evaluation metrics: Accuracy, F1 Score, ROC-AUC  
- Hyperparameter tuning applied to XGBoost for final model  

### 6. Final Submission
- Selected optimized XGBoost model  
- Generated predictions for test set  
- Constructed submission with `uniqueid`, `country`, and `bank_account`  

---

## Results Summary

| Model                 | Accuracy | F1 Score | ROC-AUC | Notes                          |
|-----------------------|---------|----------|---------|--------------------------------|
| Logistic Regression    | X       | X        | X       | Baseline                       |
| Random Forest          | X       | X        | X       | Full vs reduced features       |
| Gradient Boosting      | X       | X        | X       | Baseline outperformed reduced  |
| Extra Trees            | X       | X        | X       | Feature importance explored    |
| **XGBoost (optimized)**| X       | X        | X       | Final selection                |

---

## Key Learnings & Challenges

- Working with real-world, messy survey data requires careful preprocessing.  
- Class imbalance necessitated prioritizing F1 and ROC-AUC over accuracy.  
- Feature selection did not always improve ensemble models—tree-based (Baseline model for GradientBoost outperformed it's improved version), models handle redundancy naturally.  
- Iterative modeling and hyperparameter tuning significantly improved XGBoost performance.

---

## How to Run

1. Clone the repository: 
```
git@github.com:karanja-dave/NgaoLabs.git
```

2. Navigate to project folder:
```
cd Financial_Inclusion_Africa
```

3. Create virtual enviroment
```bash
python -m venv .venv
```

4. Activate the virtual enviroment
```bash
.venv\Scripts\activate
```
5. Install dependencies:  
```bash
pip install -r requirements.txt
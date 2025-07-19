# Credit Card Fraud Detection (XGBoost, SHAP, PyTorch)

This project implements and benchmarks machine learning models to detect fraudulent credit card transactions using a highly imbalanced real-world dataset.

**Note:** You can find the deployment process [here](https://github.com/soutogustavo/deployment-cc-fraud-detection).

## Goals

- Explore (anonymized) transactional data.
- Build a Machine Learning (ML) model to detect fraud in credit card transactions.

## Dataset

- Source: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Contains 284,807 transactions with 492 frauds (~0.17%)
- Features are anonymized via PCA (`V1` to `V28`) + `Amount` and `Time`
- Target variable: `Class` (0 = legitimate, 1 = fraud)

## Models Implemented

- Logistic Regression (baseline)
- Fully Connected Neural Network (PyTorch)
- Gradient Boosted Trees (XGBoost) **Best performance**

## Highlights

- Handles severe class imbalance using:
  - Class weighting
  - Focal loss (for NN)
  - AUPRC as primary evaluation metric
- Evaluation metrics:
  - Average Precision (AUPRC)
  - Precision, Recall, F1
- Model performance compared on validation and test sets

## Best Result (XGBoost)

- **Test AUPRC:** 0.877
- Robust to class imbalance
- Interpretable via SHAP

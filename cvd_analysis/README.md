# Cardiovascular Disease Risk Analysis Report

## 1. Introduction
This project aims to analyze the risk factors associated with Cardiovascular Disease (CVD) using machine learning techniques. We utilized the `cardio_train.csv` dataset to train various classification models and analyze the impact of specific behavioral habits, such as smoking, on heart health.

## 2. Data Selection
**Dataset Used:** `cardio_train.csv`
- **Features:** Age, Gender, Height, Weight, Systolic Blood Pressure (`ap_hi`), Diastolic Blood Pressure (`ap_lo`), Cholesterol, Glucose, Smoking, Alcohol intake, Physical Activity.
- **Target Variable:** `cardio` (presence or absence of cardiovascular disease).

## 3. Data Preprocessing
To ensure high data quality and model performance, the following preprocessing steps were applied:

1.  **Duplicate Removal**: Identified and removed duplicate entries to prevent data leakage and bias.
2.  **Feature Engineering**: 
    - Converted `age` from days to years (`age_years`) for better interpretability.
    - Dropped the `id` column as it has no predictive value.
3.  **Data Cleaning & Outlier Removal**:
    - **Blood Pressure**: Filtered for medically valid ranges:
        - Systolic (`ap_hi`): 60 - 240 mmHg
        - Diastolic (`ap_lo`): 30 - 160 mmHg
        - Enforced strict condition: `ap_hi > ap_lo`
    - **Anthropometric Data**: Removed unrealistic height (< 100cm) and weight (< 30kg) values.
4.  **Scaling**: Applied `StandardScaler` to normalize features (Mean=0, Std=1), which is crucial for distance-based algorithms like KNN and SVM.

## 4. Methodology: Model Execution
We implemented a diverse set of machine learning algorithms to solve this classification problem:

### Linear Models
- **Logistic Regression**: Used as a strong baseline for binary classification.
- **Support Vector Machine (Linear SVC)**: Effective for finding the best separating hyperplane.

### Instance-Based Models
- **K-Nearest Neighbors (KNN)**: Classifies based on feature similarity.

### Tree-Based Models
- **Decision Tree**: Provides interpretable decision rules (exported to `plots/decision_tree_rules.txt`).
- **Random Forest**: An ensemble of trees to reduce overfitting.
- **Gradient Boosting**: Builds trees sequentially to correct previous errors.
- **AdaBoost**: Focuses on difficult-to-classify instances.

### Probabilistic Models
- **Naive Bayes (GaussianNB)**: Assumes feature independence.

### Ensemble Results
- **Voting Classifier**: A hard-voting ensemble combining Logistic Regression, Random Forest, and Gradient Boosting.

## 5. Statistical Analysis: The Effect of Smoking
We performed a specific analysis on the impact of smoking:
- **Prevalence**: 
  - Non-Smokers: ~49.73% CVD presence
  - Smokers: ~46.84% CVD presence
- **Interpretation**: Interestingly, the unadjusted data showed slightly lower CVD rates in the smoking group. This is a classic case of **Simpson's Paradox** or confounding variables, largely due to age distributions (smokers in this dataset tend to be younger on average than the non-smokers who developed CVD). This highlights the importance of multivariate modeling over simple univariate statistics.

## 6. Model Performance Comparison
The models were evaluated on a held-out test set (20% of data).

| Rank | Model | Accuracy |
| :--- | :--- | :--- |
| **1** | **Gradient Boosting** | **73.53%** |
| 2 | Voting Classifier | 73.36% |
| 3 | Decision Tree | 73.19% |
| 4 | Logistic Regression | 72.97% |
| 5 | SVM | 72.97% |
| 6 | AdaBoost | 72.94% |
| 7 | Naive Bayes | 71.04% |
| 8 | Random Forest | 70.98% |
| 9 | K-Nearest Neighbors | 69.59% |

## 7. Conclusion
- **Gradient Boosting** emerged as the top-performing model, demonstrating its ability to capture complex non-linear patterns in tabular medical data.
- **Decision Trees** provided high interpretability while maintaining competitive accuracy.
- **Linear Models (LogReg, SVM)** performed surprisingly well, suggesting a strong linear component to the decision boundary.
- **Feature Importance**: Analysis from Logistic Regression coefficients identified **Systolic Blood Pressure (`ap_hi`)**, **Age**, and **Cholesterol** as the most significant risk factors for CVD.

## 8. Outputs & Artifacts
- **Results Log**: `results.txt` (Full detailed metrics)
- **Plots**:
  - `plots/feature_importance.png`
  - `plots/confusion_matrix_[Model].png`
  - `plots/smoking_impact.png`
- **Rules**: `plots/decision_tree_rules.txt`

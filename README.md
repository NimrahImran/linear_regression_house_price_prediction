# House Price Prediction — Linear Regression

An end-to-end machine learning pipeline that predicts house prices using **Linear Regression**, built on the USA Housing dataset.

## 📌 Problem Statement
- **Objective:** Predict house sale prices based on regional economic and property features
- **Dataset:** USA Housing dataset (`USA_Housing.csv`) — 5,000 records, 7 columns
- **Problem Type:** Regression
- **Target Variable:** `Price`
- **Business Relevance:** Accurate price prediction helps buyers, sellers, and real estate agents estimate fair market value and make informed pricing decisions
- **Success Metric:** R² > 0.85 → **Achieved: R² = 0.9170**

## 📊 Dataset
- **Rows:** 5,000
- **Features used:** Avg. Area Income, Avg. Area House Age, Avg. Area Number of Rooms, Avg. Area Number of Bedrooms, Area Population
- **Dropped column:** `Address` (identifier, no predictive value)
- No missing values, no duplicate rows, no constant columns detected

## 🛠 Tech Stack
- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- XGBoost
- Joblib

## 🔍 Pipeline / Methodology
1. **Data Loading & Validation** — shape, info, summary statistics
2. **Data Cleaning** — standardized column names, dropped ID-like column (`Address`), checked for duplicates and constant columns
3. **Exploratory Data Analysis** — dashboard overview, target distribution, correlation analysis, skewness check
4. **Outlier Treatment** — boxplot-based capping applied to numeric features
5. **Feature Engineering** — checked and removed highly correlated features (>0.95); none found
6. **Preprocessing Pipeline** — median imputation + standard scaling for numeric features (via `ColumnTransformer`)
7. **Train-Test Split** — 80/20 split (4,000 train / 1,000 test)
8. **Model Comparison** — benchmarked 12 regression algorithms using 5-fold cross-validation
9. **Model Training** — final pipeline trained using **Linear Regression**
10. **Model Evaluation** — MAE, RMSE, R², Adjusted R², Actual vs Predicted plot, residual distribution
11. **Sanity Checks** — train vs test score comparison to rule out data leakage
12. **Cross Validation** — 5-fold CV for robustness check
13. **Hyperparameter Tuning** — GridSearchCV / RandomizedSearchCV (Linear Regression has no tunable hyperparameters, so the base model was used as-is)

## 📈 Results

| Metric | Value |
|--------|-------|
| **R² Score (Test)** | **0.9170** |
| Adjusted R² | 0.9166 |
| MAE | 81,359.19 |
| RMSE | 101,039.05 |
| Cross-Validation Mean R² | 0.9166 (± 0.0028) |
| Train Score vs Test Score | 0.9171 vs 0.9170 (no overfitting / leakage detected) |

**Model Comparison (Top Results):**

| Model | CV R² Score |
|-------|-------------|
| Lasso | 0.9166 |
| Linear Regression | 0.9166 |
| Ridge | 0.9166 |
| Gradient Boosting | 0.9028 |
| Extra Trees | 0.8917 |

> Note: This is a **regression** problem, so evaluation uses R², MAE, and RMSE — not precision/recall/F1, which apply to classification tasks.

## 🔑 Key Insights
- **Best Model:** Lasso, Linear Regression, and Ridge all performed virtually identically (R² ≈ 0.9166), indicating a strong **linear relationship** between the features and house price
- Linear Regression was selected as the final model since it matched the top score with the simplest, most interpretable approach
- SVM performed poorly (R² ≈ 0) without kernel/parameter tuning
- Train and test scores were nearly identical (0.9171 vs 0.9170), confirming the model generalizes well with no signs of overfitting or data leakage


## 📂 Folder Structure
```
House-Price-Prediction/
│
├── data/
│   └── USA_Housing.csv
├── house-price-prediction-linear-regression.ipynb
└── README.md
```

## 👤 Author
**Nimra Muhammad Imran**

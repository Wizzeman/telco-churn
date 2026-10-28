# Telco Customer Churn Prediction

## Overview

This project focuses on predicting customer churn using the Telco Customer Churn dataset. It explores customer characteristics, service subscriptions, contract types, and billing information to build and evaluate machine learning classification models.

The project includes data preprocessing, correlation analysis, feature importance analysis, hyperparameter tuning, and model evaluation using Random Forest and XGBoost.

## Dataset

**Dataset:** `WA_Fn-UseC_-Telco-Customer-Churn.csv`

The dataset contains 7,043 customer records and information about customer demographics, subscribed services, contract types, payment methods, and charges.

The target variable is `Churn`:

- `1`: Customer churned.
- `0`: Customer did not churn.

The `customerID` column is removed before modeling.

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost

## Project Workflow

### 1. Data Exploration

- Inspected the dataset using `head()`, `info()`, and `describe()`.
- Examined column names and data types.
- Calculated the percentage of missing values in each column.
- Investigated correlations between numerical features and the target variable.

### 2. Data Preprocessing

**Removing unnecessary columns**
- Dropped `customerID`.

**Binary encoding**
- Converted `gender`, `Partner`, `Dependents`, `PhoneService`, `PaperlessBilling`, and `Churn` into numerical values.
- Mapped `Yes` to `1` and `No` to `0`.
- Mapped `Male` to `1` and `Female` to `0`.

**Service feature encoding**
- Replaced `No internet service` and `No phone service` with `No` in the relevant service columns.
- Encoded these service features as binary values.

**One-hot encoding**
- Applied `pd.get_dummies()` to `InternetService`, `Contract`, and `PaymentMethod`.
- Used `drop_first=True` and `dtype=int`.

**Handling missing values**
- Converted `TotalCharges` to a numeric data type using `pd.to_numeric()` with `errors='coerce'`.
- Replaced the resulting missing values in `TotalCharges` with `0`.

After preprocessing, the dataset contains 24 columns, including the target variable.

### 3. Correlation Analysis

Calculated feature correlations with `Churn` to examine their relationships with the target.

The strongest positive correlations observed were:

| Feature | Correlation |
|---|---:|
| InternetService_Fiber optic | 0.3080 |
| PaymentMethod_Electronic check | 0.3019 |
| MonthlyCharges | 0.1934 |
| PaperlessBilling | 0.1918 |
| SeniorCitizen | 0.1509 |

The strongest negative correlations were:

| Feature | Correlation |
|---|---:|
| tenure | -0.3522 |
| Contract_Two year | -0.3023 |
| InternetService_No | -0.2279 |
| TotalCharges | -0.1983 |
| Contract_One year | -0.1778 |

These values represent correlations, not causal relationships.

### 4. Train-Test Split and Feature Selection

- Defined `Churn` as the target variable.
- Used an 80/20 train-test split.
- Set `random_state=42`.
- Used stratification to preserve the target-class proportions.
- Examined Random Forest feature importances.
- Defined a set of 20 input features for the subsequent model experiments.

### 5. Random Forest

A Random Forest classifier was trained and evaluated.

**Hyperparameter tuning**

Used `RandomizedSearchCV` with:

- 30 parameter combinations sampled.
- 5-fold cross-validation.
- F1-score as the optimization metric.
- `random_state=42`.
- `n_jobs=-1`.

Best parameters returned by the search:

```python
{
    'n_estimators': 300,
    'max_depth': 10,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'max_features': 'log2',
    'bootstrap': True
}
```

The best mean cross-validation F1-score was **0.5642**.

The final Random Forest model was evaluated on the test set using accuracy, classification report, confusion matrix, and ROC-AUC. A ROC curve was also plotted.

### 6. XGBoost

An `XGBClassifier` was trained using the following initial parameters:

```python
XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    random_state=42,
    eval_metric="logloss"
)
```

The evaluation included:

- Accuracy
- Classification report
- Confusion matrix
- ROC-AUC
- Five-fold cross-validation using ROC-AUC scoring

The initial model's mean five-fold ROC-AUC was **0.8417**.

**Grid Search**

Used `GridSearchCV` with five-fold cross-validation and ROC-AUC scoring.

The parameter grid included:

- `n_estimators`: 100, 200, 300
- `max_depth`: 3, 4, 5
- `learning_rate`: 0.01, 0.05, 0.1

Best parameters:

```python
{
    'learning_rate': 0.05,
    'max_depth': 3,
    'n_estimators': 100
}
```

The best mean cross-validation ROC-AUC was **0.8472**.

The resulting best estimator was evaluated on the held-out test set.

## Model Evaluation Results

The following results are taken from the notebook's saved outputs.

| Metric | Random Forest | XGBoost (initial) | XGBoost (tuned) |
|---|---:|---:|---:|
| Accuracy | 0.8034 | 0.7991 | 0.80* |
| Churn Precision | 0.67 | 0.65 | 0.66 |
| Churn Recall | 0.52 | 0.53 | 0.52 |
| Churn F1-score | 0.58 | 0.58 | 0.58 |
| ROC-AUC | 0.8429 | 0.8420 | 0.8441 |

\*The tuned XGBoost accuracy is displayed as approximately 0.80 based on the classification report; the notebook does not print a separate accuracy value for this model.

### Interpretation

- Random Forest achieved an accuracy of approximately 80.34% and a ROC-AUC of 0.8429.
- The initial XGBoost model achieved an accuracy of approximately 79.91% and a ROC-AUC of 0.8420.
- After grid search, XGBoost achieved a test ROC-AUC of 0.8441.
- Churn recall was approximately 0.52–0.53 across the evaluated models, indicating that the models did not identify every customer who churned.

## Repository Contents

The notebook uses the following dataset filename:

```text
WA_Fn-UseC_-Telco-Customer-Churn.csv
```

The main analysis notebook is:

```text
Telco_churn(1).ipynb
```

For the repository, the notebook can be renamed to `Telco_churn.ipynb` if that is the filename committed to GitHub.

## How to Run

1. Clone or download the repository.
2. Install the required libraries:

   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn xgboost jupyter
   ```

3. Ensure the dataset is available at the path referenced in the notebook.
4. Launch Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

5. Open the notebook and execute its cells in order.

## Project Scope

This project covers exploratory data analysis, categorical encoding, missing-value handling, correlation analysis, feature importance, classification, hyperparameter tuning, cross-validation, and evaluation of Random Forest and XGBoost models.

The reported results are based on the experiments recorded in the notebook.

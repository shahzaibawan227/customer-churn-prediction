# Customer Churn Prediction with Tree-Based Models

## Project overview

This project develops and evaluates machine-learning models to predict
customer churn using the IBM Telco Customer Churn dataset. The workflow
covers exploratory data analysis (EDA), data cleaning, preprocessing,
model training, validation-based model development, final test
evaluation, SHAP explainability, and business reporting.

**Target:** `Churn` (`Yes` = churn, `No` = no churn).

**Objective:** Identify customers at elevated churn risk so a telecom
business can investigate retention opportunities. Predictions support
prioritization; they do not prove why a customer churns.

## Dataset

The notebook was developed in Kaggle using:

``` text
/kaggle/input/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

The dataset contained 7,043 rows and 21 original columns. `customerID`
was excluded from predictors, leaving 19 input features after separating
the target.

Target distribution: - No churn: 5,174 customers (about 73.46%) - Churn:
1,869 customers (about 26.54%)

### Data cleaning decision

`TotalCharges` was initially stored as text and contained 11
whitespace-only values. All 11 belonged to customers with `tenure == 0`.
The notebook converted the column to numeric and assigned `0.0` to those
rows, assuming zero tenure means no accumulated charges. This assumption
should be revisited if source-data definitions indicate otherwise.

## Workflow

1.  **EDA and cleaning:** inspect dimensions, data types, descriptive
    statistics, missing values, duplicates, and target balance.
2.  **Data splitting:** stratified train/validation/test split of
    60%/20%/20%, with a fixed random seed. The test set is held out from
    model and threshold selection.
3.  **Preprocessing:** numeric features use median imputation and
    standardization; categorical features use most-frequent imputation
    and one-hot encoding with `handle_unknown="ignore"`.
4.  **Models:** Logistic Regression, Decision Trees (Gini and entropy),
    Bagging, Random Forest, AdaBoost, Gradient Boosting, and XGBoost.
5.  **Evaluation:** accuracy, precision, recall, F1-score, ROC-AUC, and
    confusion matrices. Validation results guide model selection;
    selected models are then evaluated on the held-out test set.
6.  **Explainability:** SHAP summary plot for XGBoost, followed by
    cautious business interpretation.

## Final test results

The following are the reported test metrics. Classification metrics use
the default threshold of 0.50.

  Model                       Accuracy   Precision   Recall   F1-score   ROC-AUC
  ------------------------- ---------- ----------- -------- ---------- ---------
  Logistic Regression            0.806       0.657    0.559      0.604     0.842
  XGBoost                        0.807       0.680    0.516      0.587     0.847
  Random Forest                  0.801       0.659    0.516      0.579     0.838
  Gradient Boosting              0.798       0.649    0.519      0.577     0.840
  AdaBoost                       0.793       0.633    0.527      0.575     0.839
  Decision Tree (depth 2)        0.751       0.527    0.594      0.558     0.780
  Bagging                        0.781       0.607    0.500      0.548     0.813

### Interpretation

-   Logistic Regression had the highest test F1-score (0.604) and, among
    these models, the next-highest recall after the depth-2 tree.
-   XGBoost had the highest test accuracy (0.807), precision (0.680),
    and ROC-AUC (0.847).
-   The depth-2 Decision Tree had the highest recall (0.594), but lower
    precision and ROC-AUC.
-   The appropriate model depends on retention capacity and the relative
    costs of false positives and false negatives. These metrics do not
    establish financial savings or causal drivers of churn.

Metrics are rounded to three decimal places. Performance on future
customers may differ.

## SHAP findings

The supplied XGBoost SHAP summary plot ranked these features highest
among those displayed:

1.  `cat__Contract_Month-to-month`
2.  `num__tenure`
3.  `cat__InternetService_Fiber optic`

`num__MonthlyCharges` was also among the most influential displayed
features.

SHAP values indicate how features contribute to the fitted model output.
They do **not** prove that a feature causes churn. Potential
actions---reviewing contract suitability, onboarding, service quality,
or pricing---should be treated as hypotheses and validated through
further analysis or controlled experiments.

## Suggested project structure

``` text
customer-churn-prediction/
├── notebooks/
│   └── customer_churn_prediction.ipynb
├── reports/
│   └── executive_summary.docx
├── requirements.txt
├── README.md
└── .gitignore
```

Adjust filenames to match your saved files. Keep the dataset outside the
repository unless its license and sharing terms explicitly permit
redistribution. Do not commit credentials, private customer information,
or unnecessary serialized model files.

## Reproducing the project

### 1. Provide the dataset

Make the CSV available at the expected Kaggle path or update the
notebook's data-loading cell to point to your local copy.

### 2. Create a virtual environment

Python 3.13.15 was used in the reported Kaggle environment.

``` bash
python -m venv .venv
```

Activate it:

**Windows PowerShell**

``` powershell
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux**

``` bash
source .venv/bin/activate
```

### 3. Install dependencies

``` bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the notebook

Open Jupyter Notebook or JupyterLab and run all cells from top to
bottom. Confirm the dataset path exists, metrics reproduce, and the SHAP
plot is generated. Results can differ if package versions, data, or
random seeds differ.

After verifying the environment, capture exact installed versions with:

``` bash
python -m pip freeze > requirements-lock.txt
```

## Limitations and responsible use

-   The model learns patterns from historical data and cannot guarantee
    future customer behavior.
-   SHAP explains model behavior; it does not establish causation.
-   Select thresholds using validation data, operational capacity, and
    business costs---not test-set tuning.
-   Check probability calibration, subgroup performance, data drift, and
    privacy requirements before deployment.
-   Validate retention interventions experimentally before claiming
    business savings.

## Future improvements

-   Choose a classification threshold using validation data and explicit
    business costs.
-   Use cross-validation for more robust hyperparameter selection.
-   Assess calibration and precision-recall curves.
-   Evaluate subgroup performance and model drift.
-   Package the full preprocessing/model pipeline for inference and add
    input-schema validation.
-   Test retention actions through a monitored pilot and controlled
    experiments.

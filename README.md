# ❤️ Heart Disease Prediction — Advanced Machine Learning Portfolio

An end-to-end supervised machine learning project for predicting the presence of heart disease from clinical variables. This portfolio demonstrates a production-minded machine learning workflow including data auditing, leakage prevention, categorical feature handling, model benchmarking, stratified cross-validation, hyperparameter optimization, probability calibration, threshold optimization, explainability, error analysis, model stability analysis, serialization, and reusable inference.

> **⚠️ Medical Disclaimer:** This is an educational machine learning portfolio project and **not a clinical diagnostic system**. The dataset is small, and the reported performance must not be interpreted as evidence of clinical safety, medical validity, or real-world deployment readiness.

---

## 📑 Table of Contents

* [Project Overview](#-project-overview)
* [Problem Statement](#-problem-statement)
* [Project Objectives](#-project-objectives)
* [Dataset](#-dataset)
* [Data Quality and Feature Decisions](#-data-quality-and-feature-decisions)
* [Exploratory Data Analysis](#-exploratory-data-analysis)
* [Machine Learning Methodology](#-machine-learning-methodology)
* [Train-Test Strategy](#-train-test-strategy)
* [Preprocessing Pipeline](#-preprocessing-pipeline)
* [Baseline Model](#-baseline-model)
* [Model Benchmarking](#-model-benchmarking)
* [Cross-Validation](#-cross-validation)
* [Hyperparameter Optimization](#-hyperparameter-optimization)
* [Model Selection](#-model-selection)
* [Probability Calibration](#-probability-calibration)
* [Decision Threshold Optimization](#-decision-threshold-optimization)
* [Final Holdout Evaluation](#-final-holdout-evaluation)
* [Final Evaluation Metrics](#-final-evaluation-metrics)
* [Calibration Analysis](#-calibration-analysis)
* [Explainability](#-explainability)
* [SHAP Explainability](#-shap-explainability)
* [Error Analysis](#-error-analysis)
* [Model Stability Analysis](#-model-stability-analysis)
* [Model Serialization](#-model-serialization)
* [Prediction Pipeline](#-prediction-pipeline)
* [Project Structure](#-project-structure)
* [Installation](#-installation)
* [Usage](#-usage)
* [Technologies](#-technologies)
* [Reproducibility](#-reproducibility)
* [Limitations and Responsible Use](#-limitations-and-responsible-use)
* [Future Improvements](#-future-improvements)
* [Portfolio Checklist](#-portfolio-checklist)
* [Author](#-author)

---

# 🔬 Project Overview

This project develops a machine learning classification system that predicts whether a patient record is associated with heart disease.

The project is designed as an **advanced machine learning portfolio**, emphasizing not only predictive performance but also:

* Leakage prevention
* Robust preprocessing
* Model comparison
* Cross-validation
* Hyperparameter optimization
* Probability calibration
* Decision-threshold optimization
* Explainability
* Error analysis
* Model stability
* Reproducible model packaging
* Responsible use of healthcare-related machine learning

The complete workflow is:

```text
Data Audit
    ↓
Data Quality Decisions
    ↓
Exploratory Data Analysis
    ↓
Stratified Train/Test Split
    ↓
Leakage-Safe Preprocessing
    ↓
Logistic Regression Baseline
    ↓
Multi-Model Benchmarking
    ↓
5-Fold Stratified Cross-Validation
    ↓
Hyperparameter Optimization
    ↓
Candidate Model Selection
    ↓
Probability Calibration
    ↓
OOF Threshold Optimization
    ↓
Final Holdout Evaluation
    ↓
Calibration Analysis
    ↓
Error Analysis
    ↓
Permutation Importance / SHAP
    ↓
Repeated CV Stability Analysis
    ↓
Model Serialization
    ↓
Reusable Inference Pipeline
```

---

# 🎯 Problem Statement

## Machine Learning Task

The project formulates heart disease prediction as a **binary classification problem**.

The target variable is:

```text
Heart Disease
```

with the binary classes:

```text
0 → No Heart Disease
1 → Heart Disease
```

The model receives clinical and demographic variables and estimates the probability that a patient record belongs to the positive class.

---

# 🎯 Project Objectives

The main objectives are to:

1. Audit the supplied dataset.
2. Identify missing values, duplicates, and identifier columns.
3. Correctly distinguish categorical clinical codes from continuous measurements.
4. Perform exploratory data analysis.
5. Build a leakage-safe preprocessing pipeline.
6. Establish an interpretable Logistic Regression baseline.
7. Benchmark multiple machine learning algorithms.
8. Compare models using stratified cross-validation.
9. Optimize strong candidate models using randomized hyperparameter search.
10. Select a final candidate using validation evidence.
11. Calibrate predicted probabilities.
12. Optimize the classification threshold independently from model fitting.
13. Evaluate the final model on an untouched holdout set.
14. Analyze model errors.
15. Measure permutation-based feature importance.
16. Optionally generate SHAP explanations for XGBoost.
17. Assess model stability using repeated stratified cross-validation.
18. Serialize the complete inference pipeline.
19. Provide a reusable prediction function.
20. Document limitations and responsible-use considerations.

---

# 📊 Dataset

The project uses:

```text
Heart_Disease_Prediction (3).csv
```

The supplied dataset contains:

* **270 observations**
* **15 columns**
* A binary target variable: `Heart Disease`
* An identifier column: `ID`
* Continuous clinical measurements
* Coded categorical clinical variables

## Dataset Characteristics

The notebook identifies the following important characteristics:

* `Heart Disease` is the target.
* `ID` is largely missing and behaves as an identifier.
* `Sex` contains textual categories such as `Male` and `Female`.
* Several integer-valued variables represent **categorical clinical codes**, rather than continuous numerical measurements.
* The dataset contains no duplicate rows.
* Both target classes are present.

---

# 🧹 Data Quality and Feature Decisions

## Identifier Removal

The `ID` column is excluded from modeling.

The notebook identifies that `ID` contains only a small number of observed values and does not represent meaningful predictive information.

```python
ID_COLUMNS = ["ID"]
```

This prevents an identifier from accidentally becoming a model feature.

---

## Missing Values

The notebook checks missing values before modeling.

The major missingness is associated with `ID`, which is removed from the feature matrix.

Therefore, the modeling variables in the supplied dataset do not contain substantial missingness requiring external data cleaning.

Nevertheless, the modeling pipeline contains imputation steps to make the workflow robust to missing values during inference.

---

# 🧬 Feature Classification

A major design decision in this project is distinguishing **continuous variables** from **coded categorical variables**.

## Categorical Features

The following variables are treated as categorical:

```text
Sex
Chest pain type
FBS over 120
EKG results
Exercise angina
Slope of ST
Number of vessels fluro
Thallium
```

This prevents the model from incorrectly assuming that a category code such as `4` has twice the mathematical meaning of a category code such as `2`.

## Numerical Features

The continuous numerical variables are:

```text
Age
BP
Cholesterol
Max HR
ST depression
```

---

# 📈 Exploratory Data Analysis

The exploratory analysis is intentionally performed before model development while keeping the final holdout set isolated from model-selection decisions.

The notebook investigates:

### Target Distribution

The distribution of:

```text
Heart Disease = 0
Heart Disease = 1
```

is visualized to understand class balance.

### Numerical Distributions

The project examines distributions of:

* Age
* Blood pressure
* Cholesterol
* Maximum heart rate
* ST depression

### Categorical Distributions

Categorical variables are visualized against the target to inspect differences between disease and non-disease groups.

### Correlation Analysis

A correlation matrix is generated for the continuous numerical variables and the target.

### Boxplots

Numerical variables are compared across the two target classes to identify differences and potential outliers.

> EDA observations are intended to be based on the generated visualizations rather than assumptions made before analysis.

---

# 🔐 Machine Learning Methodology

The project follows a leakage-aware methodology.

A key principle is:

> **Preprocessing must be learned only from training data.**

Therefore:

* Imputation is inside the pipeline.
* Scaling is inside the pipeline.
* One-hot encoding is inside the pipeline.
* Model fitting is inside the pipeline.
* Cross-validation operates on the complete pipeline.

This prevents information from validation folds from leaking into preprocessing operations.

---

# 🔀 Train-Test Strategy

The dataset is divided using a **stratified 80/20 train-test split**.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)
```

### Training Partition

Used for:

* Model fitting
* Cross-validation
* Benchmarking
* Hyperparameter optimization
* Candidate selection
* Calibration
* Threshold selection
* Stability analysis

### Holdout Partition

The holdout set is reserved for the final evaluation.

It is **not used** for:

* Hyperparameter tuning
* Model selection
* Threshold selection
* Preprocessing fitting

This provides a more honest estimate of final generalization performance.

---

# ⚙️ Preprocessing Pipeline

The project uses a `ColumnTransformer` containing separate numerical and categorical branches.

## Numerical Pipeline

```text
Numerical Features
       ↓
Median Imputation
       ↓
StandardScaler
       ↓
Model
```

Implemented with:

```python
Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
```

## Categorical Pipeline

```text
Categorical Features
       ↓
Most-Frequent Imputation
       ↓
One-Hot Encoding
       ↓
Model
```

Implemented with:

```python
Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    ))
])
```

The use of:

```python
handle_unknown="ignore"
```

helps prevent inference failures when an unseen categorical value appears.

---

# 📏 Baseline Model

The first machine learning model is **Logistic Regression**.

Logistic Regression is selected as the baseline because it is:

* Interpretable
* Strong for binary classification
* Computationally efficient
* Probability-producing
* Compatible with standardized numerical features
* Compatible with one-hot encoded categorical features

The baseline uses:

```python
class_weight="balanced"
```

to account for potential class imbalance.

The baseline is evaluated using:

* ROC-AUC
* PR-AUC
* Classification report
* Precision
* Recall
* F1-score
* Accuracy

---

# 🤖 Model Benchmarking

The project compares six complementary model families.

| Model               | Purpose                               |
| ------------------- | ------------------------------------- |
| Logistic Regression | Interpretable classical baseline      |
| SVM                 | Margin-based nonlinear classification |
| Random Forest       | Bagged decision-tree ensemble         |
| Extra Trees         | Highly randomized tree ensemble       |
| Gradient Boosting   | Sequential boosting model             |
| XGBoost             | Advanced gradient-boosting model      |

All models are evaluated using the same preprocessing framework.

---

# 🔁 Cross-Validation

Model benchmarking uses:

```python
StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
```

This produces five stratified validation folds while preserving the target-class proportions as much as possible.

## Evaluation Metrics

The benchmark records:

* ROC-AUC
* PR-AUC
* Accuracy
* Precision
* Recall
* F1-score

### Primary Benchmark Metric

The primary model-selection metric during the initial benchmark is:

```text
ROC-AUC
```

Secondary metrics include:

```text
PR-AUC
Recall
Precision
F1
Accuracy
```

---

# 🔧 Hyperparameter Optimization

The strongest tree-based candidates are optimized using:

```python
RandomizedSearchCV
```

with stratified 5-fold cross-validation.

The holdout dataset remains untouched during this process.

---

## XGBoost Optimization

The XGBoost search explores parameters including:

```text
n_estimators
max_depth
learning_rate
subsample
colsample_bytree
min_child_weight
gamma
reg_alpha
reg_lambda
```

The search uses:

```text
40 randomized configurations
```

and optimizes:

```text
ROC-AUC
```

---

## Extra Trees Optimization

Extra Trees is tuned across:

```text
n_estimators
max_depth
min_samples_split
min_samples_leaf
max_features
```

The search evaluates:

```text
30 randomized configurations
```

using stratified 5-fold cross-validation.

---

# 🏆 Model Selection

After tuning, the project compares:

* Tuned XGBoost
* Tuned Extra Trees
* The benchmark leader

The candidate models are re-evaluated using cross-validation.

The selection considers:

* ROC-AUC
* PR-AUC
* Recall
* Precision
* F1-score

The notebook intentionally avoids selecting a model solely because it achieved the highest score on the holdout set.

For a healthcare-oriented educational project, the complete validation profile is more informative than a single metric.

---

# 🎯 Probability Calibration

High classification performance does not necessarily mean that predicted probabilities are reliable.

For example:

```text
Predicted probability = 0.80
```

should ideally correspond to approximately 80% positive outcomes among comparable predictions.

The selected model is therefore calibrated using:

```python
CalibratedClassifierCV(
    method="sigmoid",
    cv=5
)
```

Calibration is evaluated using:

* Brier score
* Calibration curve
* ROC-AUC
* PR-AUC

This separates:

### Discrimination

How well the model ranks patients according to predicted risk.

### Calibration

How well predicted probabilities correspond to observed frequencies.

---

# 🎚️ Decision Threshold Optimization

The default classification threshold:

```text
0.50
```

is not automatically assumed to be optimal.

Instead, the notebook generates **out-of-fold training probabilities** and evaluates thresholds between:

```text
0.10 → 0.90
```

using 161 threshold values.

For each threshold, the project calculates:

* Precision
* Recall
* F1
* Specificity

## Portfolio Objective

The selected threshold attempts to:

> **Maximize F1 while maintaining at least 0.80 recall when possible.**

This is particularly relevant to a screening-oriented educational experiment where missing positive cases can be important.

Most importantly, threshold selection is performed using **out-of-fold training predictions**, rather than using the holdout labels.

---

# 🧪 Final Holdout Evaluation

Only after:

* preprocessing design
* model benchmarking
* cross-validation
* hyperparameter optimization
* candidate selection
* probability calibration
* threshold selection

is the holdout set used for final evaluation.

The project reports performance at both:

```text
Threshold = 0.50
```

and:

```text
Optimized threshold
```

---

# 📊 Final Evaluation Metrics

The final evaluation includes:

### ROC-AUC

Measures ranking/discrimination across classification thresholds.

### PR-AUC

Measures precision-recall performance and is particularly useful when positive cases are less frequent.

### Accuracy

Measures the overall proportion of correct predictions.

### Precision

Measures the reliability of positive predictions.

### Recall / Sensitivity

Measures the proportion of actual positive cases identified by the model.

### Specificity

Measures the proportion of actual negative cases correctly identified.

### F1 Score

Balances precision and recall.

### Brier Score

Measures the quality of predicted probabilities.

### Confusion Matrix

Provides:

```text
True Positives
True Negatives
False Positives
False Negatives
```

---

# 📉 Calibration Analysis

The final model includes a reliability diagram using:

```python
CalibrationDisplay.from_predictions()
```

with quantile-based bins.

Calibration is analyzed separately from discrimination.

A model can have strong ROC-AUC while still producing poorly calibrated probabilities, so both aspects are evaluated.

---

# 🔍 Explainability

The project uses two explainability approaches.

## Permutation Importance

Permutation importance is calculated using the complete model pipeline.

It measures how much model performance decreases when a feature's values are randomly permuted.

Advantages:

* Model-agnostic
* Works with preprocessing pipelines
* Easy to interpret
* Suitable for tree and non-tree models

The notebook computes:

```text
importance_mean
importance_std
```

and visualizes the most influential features.

> **Important:** Feature importance indicates model dependence, not medical causation.

---

# 🧠 SHAP Explainability

The project also includes an optional SHAP analysis.

SHAP is used when the selected model is XGBoost and the fitted XGBoost estimator can be extracted from the calibrated pipeline.

The notebook uses:

```python
shap.TreeExplainer()
```

to generate a SHAP summary plot.

SHAP can provide:

* Global feature importance
* Direction of feature influence
* Individual prediction explanations

If the final selected model is not XGBoost, permutation importance remains the primary model-agnostic explanation.

---

# 🚨 Error Analysis

The project explicitly investigates model errors on the final holdout set.

Each prediction is categorized as:

```text
Correct
False Positive
False Negative
```

The analysis examines:

* Error counts
* Prediction probabilities
* False-negative cases
* False-positive cases
* Potentially difficult patient profiles

The project also visualizes prediction probabilities by error type.

## Important Consideration

Because the dataset contains only 270 observations and the holdout is small, error-analysis observations are:

> **Exploratory rather than statistically conclusive.**

---

# 📈 Model Stability Analysis

A single 5-fold cross-validation estimate can be noisy, particularly with a small dataset.

Therefore, the selected base model is evaluated using:

```python
RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=5,
    random_state=42
)
```

This produces:

```text
25 validation folds
```

across repeated stratified splits.

The stability analysis reports:

* ROC-AUC mean ± standard deviation
* PR-AUC mean ± standard deviation
* F1 mean ± standard deviation
* Recall mean ± standard deviation

This provides additional information about validation variability.

---

# 💾 Model Serialization

After final model selection, the calibrated model is retrained on the complete development partition.

The notebook packages the following information:

```python
ARTIFACT = {
    "model": final_model,
    "threshold": OPTIMAL_THRESHOLD,
    "target": TARGET,
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
    "dropped_columns": ID_COLUMNS,
    "random_state": RANDOM_STATE,
    "model_name": FINAL_MODEL_NAME,
}
```

The artifact is serialized using:

```python
joblib.dump()
```

to:

```text
heart_disease_model.joblib
```

This allows the trained model and its configuration to be reused without retraining.

---

# 🚀 Prediction Pipeline

The notebook provides a reusable function:

```python
predict_heart_disease(new_data)
```

The function:

1. Checks that all required features exist.
2. Selects the expected feature columns.
3. Generates predicted probabilities.
4. Applies the optimized threshold.
5. Returns the original features together with:

   * predicted probability
   * predicted class

### Output

```text
heart_disease_probability
heart_disease_prediction
```

This design reduces the possibility of feature-order mistakes during inference.

---

# 🛡️ Reproducibility and Integrity Checks

The project includes explicit integrity checks.

The notebook verifies that:

* The target is binary.
* The target is not included among model features.
* Identifier columns are excluded.
* Prediction probabilities fall within `[0, 1]`.
* The saved Joblib artifact can be reloaded.
* The stored threshold is valid.

The project uses:

```python
RANDOM_STATE = 42
```

throughout the workflow.

---

# 📁 Project Structure

A recommended repository structure is:

```text
Heart-Disease-Prediction/
│
├── Heart_Disease_Advanced_Professional_ML_Portfolio.ipynb
├── Heart_Disease_Prediction (3).csv
├── README.md
├── requirements.txt
│
├── artifacts/
│   └── heart_disease_model.joblib
│
└── figures/
    ├── target_distribution.png
    ├── numerical_distributions.png
    ├── categorical_distributions.png
    ├── correlation_matrix.png
    ├── model_benchmark.png
    ├── threshold_optimization.png
    ├── confusion_matrix.png
    ├── roc_curve.png
    ├── precision_recall_curve.png
    ├── calibration_curve.png
    └── permutation_importance.png
```

---

# 🛠️ Installation

## 1. Clone the Repository

```bash
git clone <your-repository-url>
cd Heart-Disease-Prediction
```

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

A suitable `requirements.txt` is:

```text
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
xgboost>=1.7.0
matplotlib>=3.7.0
seaborn>=0.12.0
shap>=0.42.0
joblib>=1.3.0
scipy>=1.10.0
jupyter>=1.0.0
```

---

# ▶️ Usage

## Jupyter Notebook

Launch Jupyter:

```bash
jupyter notebook
```

Open:

```text
Heart_Disease_Advanced_Professional_ML_Portfolio.ipynb
```

Run the notebook sequentially from the dataset audit through the final portfolio summary.

---

## Google Colab

The notebook is compatible with Google Colab.

Upload:

```text
Heart_Disease_Prediction (3).csv
```

The notebook automatically attempts to locate the dataset and provides an upload fallback when it is not found.

---

# 🔮 Example Inference

After training and serialization:

```python
import joblib

artifact = joblib.load("heart_disease_model.joblib")

model = artifact["model"]
threshold = artifact["threshold"]
```

Then inference can be performed using the project's prediction function:

```python
predictions = predict_heart_disease(new_patient_data)

display(predictions)
```

The resulting output includes the predicted probability and threshold-based classification.

---

# 🧰 Technologies

## Programming

* Python

## Data Processing

* Pandas
* NumPy

## Visualization

* Matplotlib
* Seaborn

## Machine Learning

* Scikit-learn
* XGBoost

## Explainability

* SHAP
* Permutation Importance

## Model Persistence

* Joblib

## Development Environment

* Jupyter Notebook
* Google Colab

---

# 📐 Machine Learning Design Principles

This project emphasizes several professional machine learning practices.

### Leakage Prevention

Preprocessing operations are placed inside scikit-learn pipelines.

### Stratification

Class proportions are preserved during train/test splitting and cross-validation.

### Multiple Models

Several model families are benchmarked rather than assuming one algorithm will perform best.

### Cross-Validation

Model comparison and tuning use stratified cross-validation.

### Holdout Integrity

The final holdout set remains untouched until final evaluation.

### Probability Calibration

The model's probability estimates are evaluated separately from classification performance.

### Threshold Optimization

The decision threshold is selected independently from model fitting.

### Explainability

Model behavior is examined using permutation importance and optional SHAP analysis.

### Stability

Repeated stratified cross-validation is used to estimate validation variability.

### Reproducibility

Random seeds, preprocessing, model configuration, and threshold information are preserved.

---

# ⚠️ Limitations and Responsible Use

This project has important limitations.

## Small Dataset

The dataset contains only:

```text
270 observations
```

Therefore, performance estimates can have substantial uncertainty.

## Limited Holdout Size

An 80/20 split produces a relatively small final holdout set.

Consequently, individual errors can noticeably affect reported metrics.

## Dataset Representativeness

The supplied dataset may not represent the population where a real healthcare model would eventually be used.

## Dataset-Specific Coding

Several clinical variables use coded categories whose meanings and conventions may depend on the source dataset.

## Calibration Transfer

Probability calibration observed on this dataset may not transfer to another population.

## Threshold Interpretation

The optimized threshold reflects the portfolio's selected objective:

```text
maximize F1 while maintaining at least 0.80 recall when possible
```

It is **not a clinical threshold or medical guideline**.

## Feature Importance Is Not Causality

Permutation importance and SHAP describe model behavior.

They do not establish that a clinical variable causes heart disease.

---

# 🏥 Responsible Medical AI Statement

This model must **not** be used to:

* Diagnose heart disease.
* Rule out heart disease.
* Recommend treatment.
* Replace a physician.
* Make autonomous clinical decisions.
* Determine patient eligibility for medical procedures.

A real clinical machine learning system would require substantially more validation and governance.

At minimum, this would include:

* Independent external validation
* Prospective evaluation
* Clinical expert review
* Fairness analysis
* Population-specific calibration
* Privacy and security controls
* Data-quality monitoring
* Model-drift monitoring
* Clinical governance
* Regulatory review
* Human oversight

---

# 🚀 Future Improvements

Several improvements could extend this project into a stronger research or deployment portfolio.

## 1. External Validation

Evaluate the model on an independent heart-disease dataset.

## 2. Bootstrap Confidence Intervals

Estimate uncertainty around:

* ROC-AUC
* PR-AUC
* Recall
* Precision
* F1
* Specificity

## 3. Fairness Analysis

Evaluate performance across relevant demographic groups, where appropriate and ethically justified.

## 4. Advanced Model Comparison

Additional algorithms could be evaluated, including:

* LightGBM
* CatBoost
* Neural Networks
* Calibrated SVM
* Ensemble methods

## 5. Cost-Sensitive Threshold Optimization

Instead of optimizing F1, future versions could optimize a domain-specific cost function based on the relative consequences of:

```text
False Positive
False Negative
```

## 6. Deployment API

The serialized model could be exposed through:

```text
FastAPI
```

or:

```text
Flask
```

## 7. Interactive Application

A demonstration interface could be built using:

```text
Streamlit
```

## 8. Containerization

The project could be packaged with:

```text
Docker
```

## 9. Automated Testing

Unit tests could verify:

* Feature validation
* Preprocessing
* Prediction output
* Probability ranges
* Model loading
* Threshold behavior

## 10. CI/CD

A GitHub Actions workflow could automatically:

* Install dependencies
* Run tests
* Validate the notebook
* Build deployment artifacts

---

# 📋 Portfolio Checklist

## Data

* [x] Dataset audit
* [x] Missing-value analysis
* [x] Duplicate check
* [x] Identifier review
* [x] Feature-type decisions
* [x] Target validation

## Modeling

* [x] Stratified train/test split
* [x] Leakage-safe preprocessing
* [x] Logistic Regression baseline
* [x] Multiple model families
* [x] 5-fold cross-validation
* [x] Randomized hyperparameter optimization
* [x] Candidate model selection
* [x] Probability calibration
* [x] Threshold optimization

## Evaluation

* [x] ROC-AUC
* [x] PR-AUC
* [x] Accuracy
* [x] Precision
* [x] Recall / Sensitivity
* [x] Specificity
* [x] F1-score
* [x] Brier score
* [x] Confusion matrix
* [x] ROC curve
* [x] Precision-recall curve
* [x] Calibration curve
* [x] Error analysis

## Explainability

* [x] Permutation importance
* [x] Optional SHAP analysis
* [x] Error inspection
* [x] Model stability analysis

## Engineering

* [x] Reproducible random state
* [x] Serialized model artifact
* [x] Reusable prediction function
* [x] Feature validation
* [x] Artifact reload test
* [x] Probability integrity checks
* [x] Responsible-use documentation

---

# 📌 Project Status

**Status:** Completed Advanced ML Portfolio Project

The notebook implements a complete machine learning workflow from raw clinical data through:

```text
Data Audit
→ EDA
→ Preprocessing
→ Benchmarking
→ Hyperparameter Optimization
→ Model Selection
→ Calibration
→ Threshold Optimization
→ Final Evaluation
→ Explainability
→ Stability Analysis
→ Model Serialization
→ Reusable Inference
```

The project is intended for **machine learning education, portfolio development, and experimentation**, not clinical use.

---

# 👨‍💻 Author

**Misgina Gebregergs**

Bachelor's Student — Mathematics Science
Addis Ababa University

### Interests

* Machine Learning
* Data Science
* Artificial Intelligence
* Mathematical Modeling
* Optimization
* Python
* Healthcare Machine Learning

---

# ⭐ Acknowledgment

This project demonstrates how mathematical thinking, statistical evaluation, and machine learning engineering can be combined to build a structured predictive modeling workflow.

The emphasis is not only on achieving a high score, but on developing a model that is:

> **reproducible, leakage-aware, interpretable, carefully evaluated, and responsibly documented.**

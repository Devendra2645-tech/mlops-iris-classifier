# MLOps Data Pipeline

## 1. Overview

This document describes the automated data pipeline implemented for the MLOps Iris Classifier project.

The pipeline consists of four stages:

1. Data Collection
2. Data Preprocessing
3. Feature Engineering
4. Data Validation

The pipeline is automated and managed using DVC (Data Version Control).

---

## 2. Pipeline Flow

```text
Collect
   ↓
Preprocess
   ↓
Feature Engineering
   ↓
Validate
```

---

## 3. Pipeline Stages

### 3.1 Data Collection

**Script:** `src/pipeline/collect.py`

The collection stage loads the Iris dataset using scikit-learn and stores the raw dataset.

**Input:**

* Iris dataset from `sklearn.datasets`

**Output:**

```text
data/raw/iris_raw.csv
```

**Result:**

* 150 rows collected
* 6 columns generated, including the species and collection timestamp

---

### 3.2 Data Preprocessing

**Script:** `src/pipeline/preprocess.py`

The preprocessing stage prepares the collected data for feature engineering.

The following operations are performed:

* Duplicate rows are removed.
* Numeric columns are converted to numeric format.
* Missing numeric values are replaced using the median.
* Rows with missing species values are removed.
* The `collected_at` column is removed.

**Input:**

```text
data/raw/iris_raw.csv
```

**Output:**

```text
data/processed/iris_preprocessed.csv
```

**Result:**

* 149 rows after preprocessing
* 5 columns

---

### 3.3 Feature Engineering

**Script:** `src/pipeline/features.py`

Additional features are created from the preprocessed Iris dataset.

The engineered features are:

* `sepal_area`
* `petal_area`
* `sepal_to_petal_length_ratio`
* `petal_length_bin`

**Input:**

```text
data/processed/iris_preprocessed.csv
```

**Output:**

```text
data/processed/iris_features.csv
```

**Result:**

* 149 rows
* 9 columns

---

### 3.4 Data Validation

**Script:** `src/pipeline/validate.py`

The validation stage checks whether the engineered dataset satisfies the required conditions.

The validation checks include:

* Expected columns are present.
* Missing values are checked.
* Species values are valid.
* Numeric feature ranges are checked.

**Input:**

```text
data/processed/iris_features.csv
```

**Result:**

```text
Validation PASSED: 149 rows, 9 columns, all checks satisfied
```

---

## 4. DVC Pipeline

The pipeline is defined in:

```text
dvc.yaml
```

DVC tracks dependencies and pipeline stages using:

```text
dvc.lock
```

The pipeline can be reproduced using:

```bash
python -m dvc repro
```

DVC automatically determines which stages need to be executed based on changes in dependencies and outputs.

---

## 5. Pipeline Execution

The pipeline was executed successfully using:

```bash
python -m dvc repro
```

The result showed that unchanged stages were skipped and the required validation stage was executed.

The final validation result was:

```text
Validation PASSED: 149 rows, 9 columns, all checks satisfied
```

---

## 6. Pipeline Status

The DVC pipeline status was checked using:

```bash
python -m dvc status
```

Result:

```text
Data and pipelines are up to date.
```

---

## 7. Pipeline DAG

The pipeline dependency graph was generated using:

```bash
python -m dvc dag
```

The resulting pipeline flow is:

```text
collect
   ↓
preprocess
   ↓
features
   ↓
validate
```

---

## 8. DVC Remote Storage

The generated pipeline artifacts were pushed to the configured DVC remote storage using:

```bash
python -m dvc push
```

Result:

```text
3 files pushed
```

---

## 9. Git Version Control

The pipeline files and DVC configuration were committed to Git.

Commit:

```text
5a64b0f feat: add automated data pipeline
```

Files committed include:

* `dvc.yaml`
* `dvc.lock`
* `src/pipeline/collect.py`
* `src/pipeline/preprocess.py`
* `src/pipeline/features.py`
* `src/pipeline/validate.py`
* Generated pipeline datasets

---

## 10. Summary

The automated MLOps data pipeline successfully performs:

```text
Data Collection
       ↓
Data Preprocessing
       ↓
Feature Engineering
       ↓
Data Validation
```

The pipeline is reproducible using DVC, the data artifacts are stored using DVC remote storage, and the pipeline code and configuration are version-controlled using Git.

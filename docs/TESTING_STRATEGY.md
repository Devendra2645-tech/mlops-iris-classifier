
# Testing Strategy

## 1. Objective

The objective of this practical is to test data processing functions, feature engineering, validation, model training, and model quality in the MLOps Iris classifier project.

## 2. Testing Framework

- pytest: Used to execute automated tests.
- pytest-cov: Used to measure code coverage.
- scikit-learn: Used for model training and evaluation.
- pandas: Used for data processing and testing.

## 3. Test Suite Structure

| Test File | Purpose |
|---|---|
| test_preprocess.py | Tests duplicate removal, missing-value imputation, missing target removal, and empty data |
| test_features.py | Tests feature calculations, ratios, bins, and zero-value handling |
| test_validate.py | Tests valid data and invalid data conditions |
| test_training_pipeline.py | Tests model training, predictions, and probability outputs |
| test_model_quality.py | Tests model accuracy and prediction validity |

## 4. Testing Layers

### Unit Testing
Tests individual preprocessing, feature engineering, and validation functions.

### Smoke Testing
Checks whether a model can train and produce predictions successfully.

### Model Quality Testing
Checks whether the trained model meets the minimum accuracy requirement of 90%.

## 5. Test Execution

Run all tests:

```bash
pytest -v
```

Run tests with coverage:

```bash
pytest --cov=src --cov-report=term-missing --junitxml=test-results.xml
```

## 6. Results

- Total tests executed: 14
- Tests passed: 14
- Tests failed: 0
- Overall project coverage: 20%
- Feature engineering coverage: 75%
- Preprocessing coverage: 83%
- Validation coverage: 76%

The lower overall coverage is due to additional project scripts that are not covered by this practical's test suite.

## 7. Conclusion

The automated test suite successfully verifies preprocessing, feature engineering, data validation, model training, and model quality. All 14 tests passed, demonstrating that the tested pipeline components work as expected.

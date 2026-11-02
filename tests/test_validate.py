
import pandas as pd
import pytest

from src.pipeline.validate import validate, DataValidationError


@pytest.fixture
def valid_data():
    return pd.DataFrame({
        "sepal length (cm)": [5.1],
        "sepal width (cm)": [3.5],
        "petal length (cm)": [1.4],
        "petal width (cm)": [0.2],
        "species": ["setosa"],
        "sepal_area": [17.85],
        "petal_area": [0.28],
        "sepal_to_petal_length_ratio": [3.64],
        "petal_length_bin": ["short"],
    })


def test_validate_clean_data(valid_data, tmp_path):
    input_path = tmp_path / "valid.csv"
    valid_data.to_csv(input_path, index=False)

    result = validate(str(input_path))

    assert len(result) == 1


def test_validate_missing_column(valid_data, tmp_path):
    input_path = tmp_path / "missing.csv"
    valid_data.drop(columns=["sepal_area"]).to_csv(input_path, index=False)

    with pytest.raises(DataValidationError, match="Validation failed"):
        validate(str(input_path))


def test_validate_invalid_species(valid_data, tmp_path):
    input_path = tmp_path / "invalid_species.csv"
    valid_data.loc[0, "species"] = "unknown"
    valid_data.to_csv(input_path, index=False)

    with pytest.raises(DataValidationError, match="Validation failed"):
        validate(str(input_path))


def test_validate_out_of_range_value(valid_data, tmp_path):
    input_path = tmp_path / "out_of_range.csv"
    valid_data.loc[0, "sepal length (cm)"] = 10.0
    valid_data.to_csv(input_path, index=False)

    with pytest.raises(DataValidationError, match="Validation failed"):
        validate(str(input_path))

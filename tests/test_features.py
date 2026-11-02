
import pandas as pd
import pytest

from src.pipeline.features import engineer_features


@pytest.fixture
def sample_data():
    return pd.DataFrame({
        "sepal length (cm)": [5.0, 6.0, 7.0],
        "sepal width (cm)": [4.0, 3.0, 2.0],
        "petal length (cm)": [2.0, 3.0, 6.0],
        "petal width (cm)": [1.0, 1.0, 1.0],
        "species": ["setosa", "versicolor", "virginica"],
    })


def test_engineer_features_calculates_areas(sample_data, tmp_path):
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"

    sample_data.to_csv(input_path, index=False)

    result = engineer_features(str(input_path), str(output_path))

    assert result.loc[0, "sepal_area"] == 20.0
    assert result.loc[0, "petal_area"] == 2.0


def test_engineer_features_calculates_ratio(sample_data, tmp_path):
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"

    sample_data.to_csv(input_path, index=False)

    result = engineer_features(str(input_path), str(output_path))

    assert result.loc[0, "sepal_to_petal_length_ratio"] == 2.5


def test_engineer_features_creates_bins(sample_data, tmp_path):
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"

    sample_data.to_csv(input_path, index=False)

    result = engineer_features(str(input_path), str(output_path))

    assert result.loc[0, "petal_length_bin"] == "short"
    assert result.loc[1, "petal_length_bin"] == "medium"
    assert result.loc[2, "petal_length_bin"] == "long"


def test_engineer_features_handles_zero_petal_length(tmp_path):
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"

    data = pd.DataFrame({
        "sepal length (cm)": [5.0],
        "sepal width (cm)": [3.0],
        "petal length (cm)": [0.0],
        "petal width (cm)": [0.0],
        "species": ["setosa"],
    })

    data.to_csv(input_path, index=False)

    result = engineer_features(str(input_path), str(output_path))

    assert pd.isna(result.loc[0, "sepal_to_petal_length_ratio"])

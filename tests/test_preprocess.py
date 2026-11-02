
import pandas as pd
import numpy as np
import pytest

from src.pipeline.preprocess import preprocess


@pytest.fixture
def sample_data():
    return pd.DataFrame({
        "sepal length (cm)": [5.1, 4.9, 5.1],
        "sepal width (cm)": [3.5, 3.0, 3.5],
        "petal length (cm)": [1.4, 1.4, 1.4],
        "petal width (cm)": [0.2, 0.2, 0.2],
        "species": ["setosa", "setosa", "setosa"],
        "collected_at": ["2026-01-01", "2026-01-02", "2026-01-03"],
    })


def test_preprocess_removes_duplicates(sample_data, tmp_path):
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"

    # Make the first and third rows exact duplicates
    sample_data.loc[2] = sample_data.loc[0]

    sample_data.to_csv(input_path, index=False)

    result = preprocess(str(input_path), str(output_path))

    assert len(result) == 2
    assert output_path.exists()


def test_preprocess_imputes_missing_values(sample_data, tmp_path):
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"

    sample_data.loc[0, "sepal length (cm)"] = np.nan
    sample_data.to_csv(input_path, index=False)

    result = preprocess(str(input_path), str(output_path))

    assert result["sepal length (cm)"].isna().sum() == 0


def test_preprocess_removes_missing_target(sample_data, tmp_path):
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"

    sample_data.loc[0, "species"] = np.nan
    sample_data.to_csv(input_path, index=False)

    result = preprocess(str(input_path), str(output_path))

    assert result["species"].isna().sum() == 0
    assert len(result) == 2


def test_preprocess_empty_dataframe(tmp_path):
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"

    columns = [
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)",
        "species",
    ]

    pd.DataFrame(columns=columns).to_csv(input_path, index=False)

    result = preprocess(str(input_path), str(output_path))

    assert result.empty
    assert output_path.exists()

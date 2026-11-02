
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier


def test_training_pipeline_smoke():
    # Generate a small synthetic dataset
    X, y = make_classification(
        n_samples=30,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=42,
    )

    # Train a simple model
    model = RandomForestClassifier(
        n_estimators=10,
        random_state=42,
    )

    model.fit(X, y)

    # Test predictions
    predictions = model.predict(X)

    assert len(predictions) == len(y)
    assert set(predictions).issubset(set(y))

    # Test prediction probabilities
    probabilities = model.predict_proba(X)

    assert probabilities.shape == (30, 3)
    assert np.allclose(probabilities.sum(axis=1), 1.0)

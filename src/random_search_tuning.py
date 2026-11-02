
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from scipy.stats import randint


# Load dataset
df = pd.read_csv("data/processed/iris_features.csv")

# Features and target
feature_cols = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
    "sepal_area",
    "petal_area",
    "sepal_to_petal_length_ratio",
]

X = df[feature_cols].copy()
y = df["species"].copy()

# Encode target labels
le = LabelEncoder()
y = le.fit_transform(y)

# Handle missing values
X = X.fillna(X.median(numeric_only=True))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# Random Forest model
model = RandomForestClassifier(random_state=42)

# Hyperparameter distributions
param_distributions = {
    "n_estimators": randint(50, 300),
    "max_depth": [3, 5, 10, 15, None],
    "min_samples_split": randint(2, 15),
    "max_features": ["sqrt", "log2"],
}

# Random Search with 5-fold cross-validation
random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_distributions,
    n_iter=30,
    cv=5,
    scoring="f1_macro",
    random_state=42,
    n_jobs=-1,
)

# MLflow tracking
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("iris-hyperparameter-tuning")

with mlflow.start_run(run_name="random_search_random_forest"):

    # Perform Random Search
    random_search.fit(X_train, y_train)

    best_model = random_search.best_estimator_
    best_params = random_search.best_params_
    best_score = random_search.best_score_

    # Test accuracy
    test_accuracy = best_model.score(X_test, y_test)

    # Log parameters and metrics
    mlflow.log_params(best_params)
    mlflow.log_metric("best_cv_f1_macro", best_score)
    mlflow.log_metric("test_accuracy", test_accuracy)

    # Log model
    mlflow.sklearn.log_model(best_model, "model")

    print("Random Search Results")
    print("----------------------")
    print(f"Best Parameters: {best_params}")
    print(f"Best CV F1 Macro: {best_score:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

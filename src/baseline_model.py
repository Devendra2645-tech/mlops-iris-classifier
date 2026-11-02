import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


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

# Create baseline model
model = DecisionTreeClassifier(random_state=42)

# 5-fold cross-validation
cv_scores = cross_val_score(
    model,
    X_train,
    y_train,
    cv=5,
    scoring="f1_macro",
)

cv_mean = cv_scores.mean()
cv_std = cv_scores.std()

# Train model
model.fit(X_train, y_train)

# Test prediction
y_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)

# MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("iris-hyperparameter-tuning")

with mlflow.start_run(run_name="baseline_decision_tree"):
    mlflow.log_param("model_type", "DecisionTreeClassifier")
    mlflow.log_metric("cv_f1_macro_mean", cv_mean)
    mlflow.log_metric("cv_f1_macro_std", cv_std)
    mlflow.log_metric("test_accuracy", test_accuracy)

    mlflow.sklearn.log_model(model, "model")

    print("Baseline Model Results")
    print("----------------------")
    print(f"CV F1 Macro Mean: {cv_mean:.4f}")
    print(f"CV F1 Macro Std:  {cv_std:.4f}")
    print(f"Test Accuracy:    {test_accuracy:.4f}")
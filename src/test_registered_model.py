import os
import mlflow
import mlflow.sklearn
import pandas as pd


# Connect to MLflow
mlflow.set_tracking_uri(
    os.getenv(
        "MLFLOW_TRACKING_URI",
        "http://127.0.0.1:5000"
    )
)


# Load the model from Staging
model_uri = "models:/iris-classifier-prod/Staging"

model = mlflow.sklearn.load_model(model_uri)

print("Registered model loaded successfully!")
print("Model URI:", model_uri)


# Load dataset
df = pd.read_csv("data/processed/iris_features.csv")


# Features used during training
feature_cols = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
    "sepal_area",
    "petal_area",
    "sepal_to_petal_length_ratio"
]


# Prepare sample input
X_sample = df[feature_cols].iloc[[0]]


# Predict
prediction = model.predict(X_sample)

print("\nSample input:")
print(X_sample)

print("\nPrediction:")
print(prediction)
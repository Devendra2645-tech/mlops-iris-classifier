
import mlflow


# Connect to MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Get experiment
experiment = mlflow.get_experiment_by_name(
    "iris-hyperparameter-tuning"
)

if experiment is None:
    print("Experiment not found!")
    exit()

# Retrieve all runs
runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id]
)

# Check whether runs exist
if runs.empty:
    print("No runs found!")
    exit()

# Select required columns
comparison = runs[
    [
        "tags.mlflow.runName",
        "metrics.cv_f1_macro_mean",
        "metrics.best_cv_f1_macro",
        "metrics.test_accuracy",
    ]
].copy()

# Combine baseline and tuning CV metrics
comparison["CV F1 Macro"] = comparison[
    "metrics.cv_f1_macro_mean"
].fillna(comparison["metrics.best_cv_f1_macro"])

# Rename columns
comparison = comparison.rename(
    columns={
        "tags.mlflow.runName": "Model",
        "metrics.test_accuracy": "Test Accuracy",
    }
)

# Display final comparison
comparison = comparison[
    ["Model", "CV F1 Macro", "Test Accuracy"]
]

print("\nModel Performance Comparison")
print("=" * 50)
print(comparison.to_string(index=False))

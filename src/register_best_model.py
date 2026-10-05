import os
import mlflow
from mlflow.tracking import MlflowClient


# -------------------------------------------------
# 1. Connect to MLflow server
# -------------------------------------------------

tracking_uri = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000"
)

mlflow.set_tracking_uri(tracking_uri)

client = MlflowClient()


# -------------------------------------------------
# 2. Get the experiment
# -------------------------------------------------

experiment = client.get_experiment_by_name(
    "iris-classification-baseline"
)

if experiment is None:
    raise RuntimeError("Experiment not found.")

print("Experiment ID:", experiment.experiment_id)


# -------------------------------------------------
# 3. Get all runs
# -------------------------------------------------

runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.f1_macro DESC"]
)

if not runs:
    raise RuntimeError("No runs found.")


# -------------------------------------------------
# 4. Select the best run
# -------------------------------------------------

best_run = runs[0]

best_run_id = best_run.info.run_id
best_f1 = best_run.data.metrics["f1_macro"]

print("\nBest Run ID:", best_run_id)
print("Best F1 Score:", best_f1)
print("Model Type:", best_run.data.params.get("model_type"))


# -------------------------------------------------
# 5. Register the best model
# -------------------------------------------------

model_uri = f"runs:/{best_run_id}/model"

model_name = "iris-classifier-prod"

result = mlflow.register_model(
    model_uri=model_uri,
    name=model_name
)

print("\nRegistered Model:", model_name)
print("Version:", result.version)


# -------------------------------------------------
# 6. Move model to Staging
# -------------------------------------------------

client.transition_model_version_stage(
    name=model_name,
    version=result.version,
    stage="Staging"
)

print(
    f"\nModel version {result.version} "
    f"has been moved to Staging."
)
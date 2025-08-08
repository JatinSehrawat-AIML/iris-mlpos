from mlflow.tracking import MlflowClient

# Connect to MLflow tracking server (local or remote)
client = MlflowClient()

# Get experiment by name
experiment_name = "iris-mlops"  # must match the name used in train.py
experiment = client.get_experiment_by_name(experiment_name)
experiment_id = experiment.experiment_id

# Get best run (highest accuracy)
runs = client.search_runs(
    experiment_ids=[experiment_id],
    order_by=["metrics.accuracy DESC"],
    max_results=1
)
best_run = runs[0]
best_run_id = best_run.info.run_id

print(f"Best Run ID: {best_run_id}")
print(f"Accuracy: {best_run.data.metrics['accuracy']}")

# Register the best model
model_name = "BestIrisModel"

# Create the registered model if it doesn't exist
try:
    client.create_registered_model(model_name)
except Exception:
    print(f"Model '{model_name}' already exists.")

# Create a new model version
model_uri = f"./mlruns/715266732106744842/54eff0499a694dcb8e1b220217818d29/artifacts/model"
client.create_model_version(
    name=model_name,
    source=model_uri,
    run_id=best_run_id
)

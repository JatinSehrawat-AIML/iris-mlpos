import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import mlflow
import os

# Load dataset
try:
    data = pd.read_csv("data/iris.csv")
except FileNotFoundError:
    raise Exception("Dataset not found. Please ensure data/iris.csv exists.")

# Split features and labels
X = data.drop("species", axis=1)
y = data["species"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set MLflow experiment
mlflow.set_experiment("iris-mlops")

# Start experiment run
with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)

    # Save model to disk
    os.makedirs("models", exist_ok=True)
    model_path = "models/model.pkl"
    joblib.dump(model, model_path)

    # Log model to MLflow
    mlflow.log_artifact(model_path)

    print(f"✅ Model trained with accuracy: {accuracy:.4f}")

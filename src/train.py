import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from mlflow.models.signature import infer_signature

# ==== Load dataset ====
try:
    data = pd.read_csv("data/iris.csv")
except FileNotFoundError:
    raise Exception("Dataset not found. Please ensure data/iris.csv exists.")

# Features & labels
X = data.drop("species", axis=1)
y = data["species"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# MLflow setup
mlflow.set_tracking_uri("file:./mlruns")  # local tracking
mlflow.set_experiment("iris-mlops")


# ==== Model training functions ====
def train_logistic_regression():
    with mlflow.start_run(run_name="LogisticRegression"):
        model = LogisticRegression(max_iter=200)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        # Infer signature and example
        signature = infer_signature(X_train, model.predict(X_train))
        input_example = X_train.iloc[:1]

        # Log params, metrics, and model
        mlflow.log_param("max_iter", 200)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(
            model,
            "model",
            input_example=input_example,
            signature=signature
        )

        print(f"✅ Logistic Regression accuracy: {acc:.4f}")


def train_random_forest():
    with mlflow.start_run(run_name="RandomForest"):
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        # Infer signature and example
        signature = infer_signature(X_train, model.predict(X_train))
        input_example = X_train.iloc[:1]

        # Log params, metrics, and model
        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(
            model,
            "model",
            input_example=input_example,
            signature=signature
        )

        print(f"✅ Random Forest accuracy: {acc:.4f}")


# ==== Run both models ====
if __name__ == "__main__":
    train_logistic_regression()
    train_random_forest()

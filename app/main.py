from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
import joblib
import pandas as pd
import sqlite3
from datetime import datetime
from prometheus_client import Counter, Histogram, start_http_server
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


app = FastAPI()
model = joblib.load("models/model.pkl")

# Start Prometheus metrics server (port 8001)
start_http_server(8001)

# Define Prometheus metrics
PREDICTION_COUNT = Counter(
    "prediction_requests_total", "Total predictions made"
)
PREDICTION_LATENCY = Histogram(
    "prediction_request_duration_seconds", "Prediction latency"
)


# Initialize DB and create logs table if not exists
def init_db():
    conn = sqlite3.connect("predictions.db")
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sepal_length REAL,
            sepal_width REAL,
            petal_length REAL,
            petal_width REAL,
            prediction TEXT,
            timestamp TEXT
        )
        """
    )
    conn.commit()
    conn.close()


init_db()


class IrisInput(BaseModel):
    sepal_length: float = Field(..., gt=0)
    sepal_width: float = Field(..., gt=0)
    petal_length: float = Field(..., gt=0)
    petal_width: float = Field(..., gt=0)
    target: Optional[str] = None  # Only used during retraining

    @field_validator("petal_length")
    def petal_smaller_than_sepal(cls, v, info):
        sepal_length = info.data.get("sepal_length")
        if sepal_length is not None and v > sepal_length:
            raise ValueError("petal_length should be less than sepal_length")
        return v


@app.get("/")
def root():
    return {"message": "Iris Classifier Ready"}


@app.post("/predict")
@PREDICTION_LATENCY.time()
def predict_iris(input: IrisInput):
    PREDICTION_COUNT.inc()

    df = pd.DataFrame(
        [[
            input.sepal_length,
            input.sepal_width,
            input.petal_length,
            input.petal_width
        ]],
        columns=[
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width"
        ]
    )

    prediction = model.predict(df)[0]

    # Log the prediction
    conn = sqlite3.connect("predictions.db")
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO logs (
            sepal_length,
            sepal_width,
            petal_length,
            petal_width,
            prediction,
            timestamp
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            input.sepal_length,
            input.sepal_width,
            input.petal_length,
            input.petal_width,
            str(prediction),
            datetime.now().isoformat()
        )
    )
    conn.commit()
    conn.close()

    return {"prediction": prediction}


def train_new_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"New model trained with accuracy: {accuracy}")
    return clf


@app.post("/retrain")
def retrain(data: List[IrisInput]):
    df = pd.DataFrame([d.dict() for d in data if d.target is not None])
    if df.empty:
        return {"error": "No training data with targets provided."}

    X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    y = df['target']
    new_model = train_new_model(X, y)
    joblib.dump(new_model, "models/model.pkl")

    # Update the running model in memory too
    global model
    model = new_model

    return {"status": "Model retrained successfully"}

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import sqlite3
from datetime import datetime

app = FastAPI()
model = joblib.load("models/model.pkl")

# Initialize DB and create logs table if not exists
def init_db():
    conn = sqlite3.connect("predictions.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sepal_length REAL,
            sepal_width REAL,
            petal_length REAL,
            petal_width REAL,
            prediction TEXT,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/")
def root():
    return {"message": "Iris Classifier Ready"}

@app.post("/predict")
def predict_iris(input: IrisInput):
    df = pd.DataFrame([[input.sepal_length, input.sepal_width, input.petal_length, input.petal_width]],
                      columns=["sepal_length", "sepal_width", "petal_length", "petal_width"])
    prediction = model.predict(df)[0]

    # Log the prediction
    conn = sqlite3.connect("predictions.db")
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO logs (sepal_length, sepal_width, petal_length, petal_width, prediction, timestamp)
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

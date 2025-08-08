from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()
model = joblib.load("models/model.pkl")

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
    return {"prediction": prediction}

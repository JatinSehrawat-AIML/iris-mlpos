import joblib
import pandas as pd

def load_model(path="models/model.pkl"):
    return joblib.load(path)

def predict(model, features):
    df = pd.DataFrame([features])
    return model.predict(df)[0]

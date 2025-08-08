# Iris MLOps Project

This project demonstrates a complete end-to-end MLOps pipeline for classifying Iris flowers using a machine learning model, integrating tools like DVC, MLflow, FastAPI, and GitHub Actions.

## Project Features

- ML model training with scikit-learn  
- Dataset versioning using DVC  
- Model tracking and logging with MLflow  
- REST API for inference using FastAPI  
- CI/CD setup via GitHub Actions (optional)  
- Docker-ready for containerization  

## Project Structure

``` bash 
iris/
├── .dvc/                  # DVC-related files for data/version control
├── .github/               # GitHub workflows (CI/CD actions etc.)
├── app/                   # Application interface or deployment-related code
├── data/                  # Raw and processed datasets
├── mlflow_logs/           # MLflow experiment logs
├── mlruns/                # MLflow tracking run data
├── models/                # Trained model artifacts
├── src/                   # Core source code for training and inference
│   ├── predict.py         # Prediction/inference logic
│   ├── train.py           # Model training script
│   ├── utils.py           # Utility functions
├── venv/                  # Python virtual environment (usually not committed)
├── .dvcignore             # DVC ignore file (like .gitignore)
├── .gitignore             # Git ignore rules
├── Dockerfile             # Docker setup for containerizing the project
├── dvc.yaml               # DVC pipeline stages and dependencies
├── README.md              # Project overview and documentation
├── requirements.txt       # Python dependencies
```
## Setup Instructions

1. Clone and Install Dependencies

```bash
git clone <your-repo-url>
cd iris-mlops-project
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Download the Dataset

```bash
mkdir -p data
curl -o data/iris.csv https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv
```

3. Version the Data (Optional but recommended)

```bash
dvc init
dvc add data/iris.csv
git add .dvc data/.gitignore data/iris.csv.dvc
git commit -m "Add dataset with DVC"
```

4. Train the Model

```bash
python src/train.py
```
<img width="370" height="156" alt="Screenshot 2025-07-25 at 19 23 22" src="https://github.com/user-attachments/assets/859e85e6-bf5b-42a6-8c28-7708fe482893" />

Output:
- Trains the model  
- Saves model as models/model.pkl  
- Logs accuracy and model in MLflow  

5. Run the FastAPI App

```bash
uvicorn app.main:app --reload
```

Open in browser:  
http://127.0.0.1:8000  
<img width="656" height="259" alt="Screenshot 2025-07-25 at 19 24 35" src="https://github.com/user-attachments/assets/85ff59dd-fc60-40b4-a182-466b112fe864" />

Swagger docs:  
http://127.0.0.1:8000/docs

<img width="500" height="500" alt="Screenshot 2025-07-25 at 19 25 10" src="https://github.com/user-attachments/assets/62c599a3-d3a6-45c8-8dce-6761511fc823" />
<img width="500" height="500" alt="Screenshot 2025-07-25 at 19 25 30" src="https://github.com/user-attachments/assets/d6df11ef-6f79-4b79-9644-93a9e879e1cd" />
<img width="500" height="500" alt="Screenshot 2025-07-25 at 19 25 37" src="https://github.com/user-attachments/assets/877d0a27-f3eb-4c67-9338-02405c2604ea" />
<img width="500" height="500" alt="Screenshot 2025-07-25 at 19 25 52" src="https://github.com/user-attachments/assets/66b603fd-6c16-4ff1-b71b-938dd3430f50" />

6. Make Predictions

Use the Swagger UI to call the /predict endpoint with:

```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

Expected response:

```json
{
  "prediction": "setosa"
}
```

7. Launch MLflow Dashboard

```bash
mlflow ui
```
<img width="737" height="271" alt="Screenshot 2025-07-25 at 19 27 52" src="https://github.com/user-attachments/assets/2e892ffb-28dd-49c2-96ad-a101ba13c76e" />



Open in browser:  
http://localhost:5000

<img width="700" height="700" alt="Screenshot 2025-07-25 at 19 28 31" src="https://github.com/user-attachments/assets/d8bd1231-86df-4a35-b95e-09a5b0835f9b" />

<img width="700" height="700" alt="Screenshot 2025-07-25 at 19 28 40" src="https://github.com/user-attachments/assets/ef24b195-eed1-4841-9232-c1420b8da94e" />

<img width="700" height="700" alt="Screenshot 2025-07-25 at 19 29 13" src="https://github.com/user-attachments/assets/89af2532-37c4-4863-8720-7f28071e02e9" />


## Tools Used

- scikit-learn for model training  
- joblib for model serialization  
- MLflow for experiment tracking  
- FastAPI for RESTful API  
- DVC for dataset versioning  
- Docker for containerization  
- GitHub Actions for CI/CD  

## For Research Paper

You can include this project in your research to demonstrate:  
- End-to-end machine learning lifecycle  
- Reproducibility and dataset tracking with DVC  
- Model performance tracking with MLflow  
- Real-time predictions using a REST API  
- Automation of retraining and deployment  


# iris-mlpos

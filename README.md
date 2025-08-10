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
├── .dvc/                    # DVC-related files for data/version control
├── .github/                 # GitHub workflows (CI/CD actions etc.)
├── app/                     # Application interface or deployment-related code
├── data/                    # Raw and processed datasets
├── mlruns/                  # MLflow tracking run data
├── models/                  # Trained model artifacts
├── src/                     # Core source code for training and inference
│   ├── predict.py           # Prediction/inference logic
│   ├── select_best_model.py # Select best model and register in MLflow
│   ├── train.py             # Model training script
│   ├── utils.py             # Utility functions
├── venv/                    # Python virtual environment (usually not committed)
├── .dvcignore               # DVC ignore file (like .gitignore)
├── .gitignore               # Git ignore rules
├── Dockerfile               # Docker setup for containerizing the project
├── dvc.yaml                 # DVC pipeline stages and dependencies
├── predictions.db           # Log incoming prediction requests and model outputs
├── README.md                # Project overview and documentation
├── requirements.txt         # Python dependencies
```
## Setup Instructions

1. Clone and Install Dependencies

```bash
git clone <your-repo-url>
cd iris-mlops
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

Output:
- Trains the model  
- Saves model as models/model.pkl  
- Logs accuracy and model in MLflow  

5. Run the FastAPI App

```bash
uvicorn app.main:app --reload
```

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

##  Tools & Technologies

Machine Learning: scikit-learn, joblib
MLOps: DVC, MLflow
API Framework: FastAPI
Containerization: Docker
Automation: GitHub Actions
Version Control: Git

## Workflow Diagram

graph LR
A[Data Collection] --> B[DVC Versioning]
B --> C[Model Training - train.py]
C --> D[MLflow Logging]
D --> E[Model Storage - models/model.pkl]
E --> F[FastAPI Deployment]
F --> G[User Prediction Requests]


## Research/Academic Use

This project can be used to demonstrate:
Complete ML lifecycle from data ingestion to deployment
Reproducibility via dataset and model versioning
Experiment tracking for research comparisons
Real-time inference with API endpoints
Automated CI/CD pipelines for ML models


## This project is licensed under the MIT License – feel free to use and modify.

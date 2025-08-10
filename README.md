🌸 Iris MLOps Pipeline
An end-to-end MLOps implementation for classifying Iris flowers, showcasing model training, version control, experiment tracking, deployment, and monitoring using modern DevOps and ML tools.

🚀 Key Features
Model Training: scikit-learn RandomForest classifier

Data Versioning: DVC for reproducible datasets

Experiment Tracking: MLflow for metrics, parameters, and artifacts

Model Serving: FastAPI REST API for real-time predictions

Automation: GitHub Actions for CI/CD workflows

Containerization: Docker for consistent deployment environments

📂 Project Structure
bash
Copy
Edit
iris-mlops/
├── .dvc/                  # DVC metadata for data versioning
├── .github/               # GitHub Actions workflows
├── app/                   # FastAPI application for inference
├── data/                  # Raw and processed data
├── mlflow_logs/           # MLflow experiment log storage
├── mlruns/                # MLflow tracking data
├── models/                # Trained model artifacts
├── src/                   # Source code
│   ├── train.py           # Model training
│   ├── predict.py         # Prediction logic
│   ├── utils.py           # Helper functions
├── Dockerfile             # Docker build configuration
├── dvc.yaml               # DVC pipeline configuration
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
⚙️ Setup Instructions
1️⃣ Clone the Repository & Install Dependencies
bash
Copy
Edit
git clone https://github.com/JatinSehrawat-AIML/iris-mlpos.git
cd iris-mlpos
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
2️⃣ Download & Version the Dataset
bash
Copy
Edit
mkdir -p data
curl -o data/iris.csv https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv

# Version with DVC
dvc init
dvc add data/iris.csv
git add data/iris.csv.dvc .dvc .gitignore
git commit -m "Add dataset with DVC"
3️⃣ Train the Model
bash
Copy
Edit
python src/train.py
✅ Saves the model to models/model.pkl
✅ Logs results to MLflow

4️⃣ Serve the Model with FastAPI
bash
Copy
Edit
uvicorn app.main:app --reload
Access the API at: http://127.0.0.1:8000
Swagger UI: http://127.0.0.1:8000/docs

5️⃣ Make Predictions via API
Example request:

json
Copy
Edit
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
Example response:

json
Copy
Edit
{
  "prediction": "setosa"
}
6️⃣ Monitor Experiments in MLflow
bash
Copy
Edit
mlflow ui
Open: http://127.0.0.1:5000

🛠 Tools & Technologies
Machine Learning: scikit-learn, joblib

MLOps: DVC, MLflow

API Framework: FastAPI

Containerization: Docker

Automation: GitHub Actions

Version Control: Git

📊 Workflow Diagram
mermaid
Copy
Edit
graph LR
A[Data Collection] --> B[DVC Versioning]
B --> C[Model Training - train.py]
C --> D[MLflow Logging]
D --> E[Model Storage - models/model.pkl]
E --> F[FastAPI Deployment]
F --> G[User Prediction Requests]
📌 Research/Academic Use
This project can be used to demonstrate:

Complete ML lifecycle from data ingestion to deployment

Reproducibility via dataset and model versioning

Experiment tracking for research comparisons

Real-time inference with API endpoints

Automated CI/CD pipelines for ML models

📄 License
This project is licensed under the MIT License – feel free to use and modify.

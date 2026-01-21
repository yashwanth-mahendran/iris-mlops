# Iris MLOps Project

End-to-end MLOps pipeline for Iris classification with DVC and MLflow tracking.

## Setup Instructions

### 1. Environment Setup
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install scikit-learn joblib mlflow "dvc[s3]" fastapi uvicorn
```

### 2. DVC Setup
```bash
# Initialize DVC
dvc init

# Add S3 remote for data versioning
dvc remote add -d s3remote s3://mlops-dvc-yash0707-prd

# Track data with DVC (if you have data/iris.csv)
dvc add data/iris.csv
dvc push
```

### 3. MLflow Setup
```bash
# Start MLflow server
mlflow server --host 127.0.0.1 --port 5000

# Access MLflow UI at: http://127.0.0.1:5000
```

## Running the Project

### Train Model
```bash
python train.py
```

### Start FastAPI Server
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### Docker Deployment
```bash
# Build Docker image
docker build -t iris-mlops .

# Run container
docker run -p 8000:8000 iris-mlops
```

### API Endpoints
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Prediction**: POST http://localhost:8000/predict

### Example API Usage
```bash
# Using curl
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "sepal_length": 5.1,
       "sepal_width": 3.5,
       "petal_length": 1.4,
       "petal_width": 0.2
     }'

# Response:
# {"prediction": "Setosa", "confidence": 0.99}
```

### Make Predictions (CLI)
```bash
python predict.py
```

### Register Model in MLflow
1. Go to http://127.0.0.1:5000
2. Click on your training run
3. Navigate to "Artifacts" → "iris_model.pkl"
4. Click "Register Model"
5. Name it "iris-classifier"

## Project Structure
```
iris-mlops/
├── train.py          # Model training with MLflow logging
├── predict.py        # Model inference with MLflow logging
├── app.py            # FastAPI application for model serving
├── Dockerfile        # Docker container configuration
├── requirements.txt  # Python dependencies
├── iris_model.pkl    # Trained model artifact
├── data/             # Data directory (DVC tracked)
└── readme.md         # This file
```
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

### 4. KServe Setup (Optional - for Kubernetes deployment)
```bash
# Install KServe on Kubernetes cluster
# Note: Requires cluster with sufficient resources (4GB+ RAM per node)

# 0. Install cert-manager (required for KServe)
helm repo add jetstack https://charts.jetstack.io
helm repo update

# Install cert-manager
helm install cert-manager jetstack/cert-manager \
  --namespace cert-manager \
  --create-namespace \
  --version v1.13.0 \
  --set installCRDs=true

# Verify cert-manager installation
kubectl get pods -n cert-manager

# 1. Install Istio using Helm with reduced resource requirements
helm repo add istio https://istio-release.storage.googleapis.com/charts
helm repo update

# Create istio-system namespace
kubectl create namespace istio-system

# Install Istio base
helm install istio-base istio/base -n istio-system

# Install Istio discovery with reduced resources
helm install istiod istio/istiod -n istio-system --wait \
  --set pilot.resources.requests.memory=128Mi \
  --set pilot.resources.requests.cpu=100m \
  --set pilot.resources.limits.memory=512Mi \
  --set pilot.resources.limits.cpu=500m

# 2. Install Knative Serving
kubectl apply -f https://github.com/knative/serving/releases/download/knative-v1.11.0/serving-crds.yaml
kubectl apply -f https://github.com/knative/serving/releases/download/knative-v1.11.0/serving-core.yaml
kubectl apply -f https://github.com/knative/net-istio/releases/download/knative-v1.11.0/net-istio.yaml

# 3. Install KServe
kubectl apply -f https://github.com/kserve/kserve/releases/download/v0.11.0/kserve.yaml
kubectl apply -f https://github.com/kserve/kserve/releases/download/v0.11.0/kserve-runtimes.yaml

# Troubleshooting: If nodes have insufficient memory
# Check node resources:
kubectl describe nodes
kubectl top nodes

# Remove taints from master node (if single-node cluster):
kubectl taint nodes --all node-role.kubernetes.io/control-plane-

# Verify installation
kubectl get pods -n cert-manager
kubectl get pods -n istio-system
kubectl get pods -n kserve-system
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

### KServe Deployment (Kubernetes)
```bash
# Prerequisites: KServe installed in cluster

# Option 1: Use Docker Hub (requires authentication)
# Login to Docker Hub
docker login

# Tag and push to your Docker Hub account
docker build -t iris-mlops .
docker tag iris-mlops your-dockerhub-username/iris-mlops:latest
docker push your-dockerhub-username/iris-mlops:latest

# Update image in deploy/inference-service.yaml to:
# image: your-dockerhub-username/iris-mlops:latest

# Option 2: Use local registry (for development)
# Start local registry
docker run -d -p 5000:5000 --name registry registry:2

# Tag and push to local registry
docker build -t iris-mlops .
docker tag iris-mlops localhost:5000/iris-mlops:latest
docker push localhost:5000/iris-mlops:latest

# Update image in deploy/inference-service.yaml to:
# image: localhost:5000/iris-mlops:latest

# Option 3: Load image directly to cluster (kind/minikube)
# For kind cluster:
kind load docker-image iris-mlops:latest

# For minikube:
minikube image load iris-mlops:latest

# Update image in deploy/inference-service.yaml to:
# image: iris-mlops:latest
# imagePullPolicy: Never

# Deploy to cluster
kubectl apply -f deploy/pvc.yaml
kubectl apply -f deploy/inference-service.yaml
kubectl apply -f deploy/service.yaml

# Check deployment status
kubectl get inferenceservice iris-classifier
kubectl get pods
kubectl describe inferenceservice iris-classifier
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
├── deploy/           # Kubernetes deployment manifests
│   ├── inference-service.yaml  # KServe InferenceService
│   ├── pvc.yaml                # PersistentVolumeClaim
│   └── service.yaml            # Kubernetes Service
├── iris_model.pkl    # Trained model artifact
├── data/             # Data directory (DVC tracked)
└── readme.md         # This file
```
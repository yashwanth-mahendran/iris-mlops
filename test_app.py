import pytest
import numpy as np
import joblib
import os
from app import app
from fastapi.testclient import TestClient

client = TestClient(app)

def test_model_exists():
    """Test that model file exists after training"""
    assert os.path.exists("iris_model.pkl"), "Model file should exist"

def test_model_prediction():
    """Test model prediction functionality"""
    if os.path.exists("iris_model.pkl"):
        model = joblib.load("iris_model.pkl")
        sample = np.array([[5.1, 3.5, 1.4, 0.2]])
        prediction = model.predict(sample)
        assert prediction[0] in [0, 1, 2], "Prediction should be 0, 1, or 2"

def test_api_health():
    """Test API health endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_api_predict():
    """Test API prediction endpoint"""
    test_data = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    response = client.post("/predict", json=test_data)
    assert response.status_code == 200
    result = response.json()
    assert "prediction" in result
    assert "confidence" in result
    assert result["prediction"] in ["Setosa", "Versicolor", "Virginica"]
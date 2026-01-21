from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Iris Classification API", version="1.0.0")

# Load model at startup
model = joblib.load("iris_model.pkl")
class_names = ["Setosa", "Versicolor", "Virginica"]

class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float

@app.get("/")
def root():
    return {"message": "Iris Classification API"}

@app.post("/predict", response_model=PredictionResponse)
def predict(features: IrisFeatures):
    # Convert to numpy array
    sample = np.array([[
        features.sepal_length,
        features.sepal_width,
        features.petal_length,
        features.petal_width
    ]])
    
    # Make prediction
    prediction = model.predict(sample)[0]
    probabilities = model.predict_proba(sample)[0]
    confidence = float(max(probabilities))
    
    return PredictionResponse(
        prediction=class_names[prediction],
        confidence=confidence
    )

@app.get("/health")
def health():
    return {"status": "healthy"}
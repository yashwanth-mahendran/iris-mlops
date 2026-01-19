# predict.py

import joblib
import numpy as np

def main():
    # Load model
    model = joblib.load("iris_model.pkl")
    print("Model loaded successfully")

    # Example input (sepal length, sepal width, petal length, petal width)
    # You can change these values
    sample = np.array([[5.1, 3.5, 1.4, 0.2]])

    # Predict
    prediction = model.predict(sample)[0]

    class_names = ["Setosa", "Versicolor", "Virginica"]
    print("Prediction:", class_names[prediction])

if __name__ == "__main__":
    main()

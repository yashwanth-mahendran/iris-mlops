# train.py

import joblib
import mlflow
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def main():
    # Set MLflow tracking URI
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("iris-mlops")
    
    with mlflow.start_run():
        # Load data
        iris = load_iris()
        X, y = iris.data, iris.target
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Model parameters
        max_iter = 200
        random_state = 42
        
        # Log parameters
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("test_size", 0.2)
        
        # Train model
        model = LogisticRegression(max_iter=max_iter, random_state=random_state)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        
        # Save model
        joblib.dump(model, "iris_model.pkl")
        mlflow.log_artifact("iris_model.pkl")
        
        print(f"Model trained with accuracy: {accuracy:.4f}")
        print("Model saved as iris_model.pkl")

if __name__ == "__main__":
    main()
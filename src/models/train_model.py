import os
import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
import numpy as np
import logging
import os
import bentoml

def main():
    logger = logging.getLogger(__name__)
    logger.info('training model')

    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path, '../../data/processed')

    X_train = pd.read_csv(file_path + "/X_train.csv")
    y_train = pd.read_csv(file_path + "/y_train.csv")
    X_test = pd.read_csv(file_path + "/X_test.csv")
    y_test = pd.read_csv(file_path + "/y_test.csv")

    # Entraînement du modèle
    
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Évaluer le modèle
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f"Train score: {train_score:.3f}")
    print(f"Test score: {test_score:.3f}")
    
    y_pred = model.predict(X_test)
    
    metrics = {
        "mse": mean_squared_error(y_test, y_pred),
        "mae": mean_absolute_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred)
    }
    
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    metrics["r2_cv_mean"] = np.mean(cv_scores)
    metrics["r2_cv_std"] = np.std(cv_scores)
    
    print(metrics)

    scaler = joblib.load(os.path.join(base_path, '../../data/processed/scaler.joblib'))

    bentoml.sklearn.save_model(
    "admission_model",
    model,
    custom_objects={             # On enregistre le scaler avec le modèle
        "scaler": scaler
    }
)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    
    main()
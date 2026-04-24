import os
import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
import numpy as np
import json
import logging

def main():
    logger = logging.getLogger(__name__)
    logger.info('evaluating model')
    # Charger les données
    X_train = pd.read_pickle("data/processed_data/X_train_scaled.pickle")
    y_train = pd.read_csv("data/processed_data/y_train.csv")
    X_test = pd.read_pickle("data/processed_data/X_test_scaled.pickle")
    y_test = pd.read_csv("data/processed_data/y_test.csv")


    # Charger les meilleurs hyperparamètres
    model = joblib.load("models/best_model.pkl")

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
    
    predictions_df = X_test
    predictions_df["y_true"] = y_test
    predictions_df["y_pred"] = y_pred
    
    predictions_df.to_csv("data/predictions.csv", index=False)
    print("Prédictions sauvegardées dans data/predictions.csv")
    
    with open("metrics/scores.json", "w") as f:
        json.dump(metrics, f, indent=4)
    print("Métriques sauvegardées dans metrics/scores.json")

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
import os
import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
import logging

def main():
    logger = logging.getLogger(__name__)
    logger.info('training model')
    X_train = pd.read_pickle("data/processed_data/X_train_scaled.pickle")
    y_train = pd.read_csv("data/processed_data/y_train.csv")
    
    
    best_params = joblib.load("models/best_params.pkl")
    
    print("Hyperparamètres chargés :", best_params)
    
    model = LinearRegression(**best_params)
    model.fit(X_train, y_train)
    
    joblib.dump(model, "models/best_model.pkl")
    
    print("Modèle entraîné sauvegardé dans models/best_model.pkl")

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    
    main()
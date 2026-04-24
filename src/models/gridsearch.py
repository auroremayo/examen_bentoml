from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
import pandas as pd
import joblib
import logging

def main():
    logger = logging.getLogger(__name__)
    logger.info('searching for best params with GridSearchCV')
    X_train = pd.read_pickle("data/processed_data/X_train_scaled.pickle")
    y_train =  pd.read_csv("data/processed_data/y_train.csv")
    X_test = pd.read_pickle("data/processed_data/X_test_scaled.pickle")
    y_test =  pd.read_csv("data/processed_data/y_test.csv")
    
    param_grid = {
        'fit_intercept': [True, False],
        'positive': [True, False]
    }
    
    
    grid_search = GridSearchCV(estimator=LinearRegression(),
                               param_grid=param_grid,
                               cv=5,
                               scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    
    # Report best score and parameters
    print(f"Best score: {grid_search.best_score_:.3f}")
    print(f"Best parameters: {grid_search.best_params_}")
    
    # Evaluate on test set
    best_model = grid_search.best_estimator_
    test_score = best_model.score(X_test, y_test)
    print(f"Test set score: {test_score:.3f}")
    
    # Sauvegarder uniquement les hyperparamètres
    joblib.dump(grid_search.best_params_, "models/best_params.pkl")
    
    print("Hyperparamètres sauvegardés dans models/best_params.pkl")

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    
    main()
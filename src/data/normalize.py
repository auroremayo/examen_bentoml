from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
import pandas as pd
import pickle
import logging

def main():
    logger = logging.getLogger(__name__)
    logger.info('normalizing data set')
    X_train = pd.read_csv("data/processed_data/X_train.csv", header=0, sep=",")
    X_test = pd.read_csv("data/processed_data/X_test.csv", header=0, sep=",")
    
    scaler1 = MinMaxScaler()
    scaler2 = RobustScaler()
    scaler3 = StandardScaler()
    
    X_train_scaled = X_train
    X_test_scaled = X_test
    
    # Colonnes 0 et 1
    X_train_scaled.iloc[:, [0,1]] = scaler1.fit_transform(X_train.iloc[:, [0,1]])
    X_test_scaled.iloc[:, [0,1]] = scaler1.transform(X_test.iloc[:, [0,1]])
    
    # Colonne 2
    X_train_scaled.iloc[:, [2]] = scaler2.fit_transform(X_train.iloc[:, [2]])
    X_test_scaled.iloc[:, [2]] = scaler2.transform(X_test.iloc[:, [2]])
    
    # Colonnes 3 à fin
    X_train_scaled.iloc[:, 3:] = scaler3.fit_transform(X_train.iloc[:, 3:])
    X_test_scaled.iloc[:, 3:] = scaler3.transform(X_test.iloc[:, 3:])
    
    X_train_scaled.to_pickle("data/processed_data/X_train_scaled.pickle")
    X_test_scaled.to_pickle("data/processed_data/X_test_scaled.pickle")

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
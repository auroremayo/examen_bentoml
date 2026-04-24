import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

df = pd.read_csv('../../data/raw/admission.csv')

df = df.drop(columns=['Serial No.'])

X = df.drop(columns=['Chance of Admit'])
y = df['Chance of Admit']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled.to_csv("../../data/processed/X_train.csv", index=False)
X_test_scaled.to_csv("../../data/processed/X_test.csv", index=False)
y_train.to_csv("../../data/processed/y_train.csv", index=False)
y_test.to_csv("../../data/processed/y_test.csv", index=False)


joblib.dump(scaler, "../../data/processed/scaler.joblib")
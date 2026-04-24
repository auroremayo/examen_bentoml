import requests
import pytest
import jwt
import time
from datetime import datetime, timedelta, timezone


BASE_URL = "http://127.0.0.1:3001"
JWT_SECRET_KEY = "admission_secret_key_2026" # Doit être la même que dans service.py
JWT_ALGORITHM = "HS256"


@pytest.fixture
def valid_token():
    payload = {"credentials": {"username": "admin", "password": "bentoml_2026"}}
    response = requests.post(f"{BASE_URL}/login", json=payload)
    return response.json().get("access_token")


# --- Tests Authentification JWT ---
def test_predict_no_token():
    response = requests.post(f"{BASE_URL}/predict", json={"input_data": {}})
    assert response.status_code == 401

def test_predict_invalid_token():
    headers = {"Authorization": "Bearer fake_token"}
    response = requests.post(f"{BASE_URL}/predict", headers=headers, json={"input_data": {}})
    assert response.status_code == 401

def test_auth_fails_if_token_expired():
    # On génère un jeton qui a expiré il y a 10 minutes
    exp = datetime.now(timezone.utc) - timedelta(minutes=10)
    expired_payload = {"sub": "admin", "exp": exp}
    expired_token = jwt.encode(expired_payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    
    headers = {"Authorization": f"Bearer {expired_token}"}
    response = requests.post(f"{BASE_URL}/predict", headers=headers, json={"input_data": {}})
    
    assert response.status_code == 401
    assert "expired" in response.json().get("detail", "").lower()

# --- Tests API Connexion ---
def test_login_success():
    payload = {"credentials": {"username": "admin", "password": "bentoml_2026"}}
    response = requests.post(f"{BASE_URL}/login", json=payload)
    assert response.status_code == 200
    assert "access_token" in response.json()

def test_login_invalid_credentials():
    payload = {"credentials": {"username": "wrong", "password": "wrong"}}
    response = requests.post(f"{BASE_URL}/login", json=payload)
    assert response.status_code == 401


# --- Tests API Prédiction ---
def test_predict_success(valid_token):
    headers = {"Authorization": f"Bearer {valid_token}"}
    data = {
        "input_data": {
            "GRE Score": 207,
            "TOEFL Score": 125,
            "University Rating": 2,
            "SOP": 2.5,
            "LOR": 2.0,
            "CGPA": 10.94,
            "Research": 1
        }
    }
    response = requests.post(f"{BASE_URL}/predict", headers=headers, json=data)
    assert response.status_code == 200
    assert "chance_of_admit" in response.json()

def test_predict_invalid_data(valid_token):
    headers = {"Authorization": f"Bearer {valid_token}"}
    # Envoi de données incomplètes
    data = {"input_data": {"GRE Score": "Pas un nombre"}}
    response = requests.post(f"{BASE_URL}/predict", headers=headers, json=data)
    assert response.status_code == 422 # Erreur de validation Pydantic
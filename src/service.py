import numpy as np
import bentoml
import jwt
from pydantic import BaseModel, Field
from starlette.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from datetime import datetime, timedelta


JWT_SECRET_KEY = "admission_secret_key_2026"
JWT_ALGORITHM = "HS256"


USERS = {
    "admin": "bentoml_2026",
    "student_user": "admission_pass"
}


class Credentials(BaseModel):
    username: str
    password: str

class AdmissionInput(BaseModel):
    gre_score: float = Field(alias="GRE Score")
    toefl_score: float = Field(alias="TOEFL Score")
    university_rating: float = Field(alias="University Rating")
    sop: float = Field(alias="SOP")
    lor: float = Field(alias="LOR")
    cgpa: float = Field(alias="CGPA")
    research: int = Field(alias="Research")


class JWTAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        if request.url.path.endswith("/predict"):
            auth_header = request.headers.get("Authorization")
            if not auth_header or not auth_header.startswith("Bearer "):
                return JSONResponse(status_code=401, content={"detail": "Token manquant ou format invalide"})
            
            try:
                token = auth_header.split(" ")[1]
                payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
                request.state.user = payload.get("sub")
            except jwt.ExpiredSignatureError:
                return JSONResponse(status_code=401, content={"detail": "Le token a expiré"})
            except jwt.InvalidTokenError:
                return JSONResponse(status_code=401, content={"detail": "Token invalide"})
        
        return await call_next(request)



@bentoml.service
class AdmissionModelService:
    """Service interne gérant l'inférence du modèle"""
    def __init__(self):
        # Chargement du modèle et du scaler depuis le Model Store
        # Assurez-vous d'avoir utilisé ce nom lors de la sauvegarde
        self.bento_model = bentoml.sklearn.get("admission_model:rc5sdar76k6gkcvh")
        self.model = self.bento_model.load_model()
        self.scaler = self.bento_model.custom_objects["scaler"]

    @bentoml.api
    def run_inference(self, features: list[float]) -> float:
        # Transformation par le scaler puis prédiction
        features_arr = np.array(features).reshape(1, -1)
        scaled_features = self.scaler.transform(features_arr)
        prediction = self.model.predict(scaled_features)
        return float(prediction[0])

@bentoml.service
class AdmissionApiService:
    """Service public exposant l'API et gérant l'authentification"""
    model_service = bentoml.depends(AdmissionModelService)

    @bentoml.api(route="/login")
    def login(self, credentials: Credentials) -> dict:
        if USERS.get(credentials.username) == credentials.password:
            expiration = datetime.utcnow() + timedelta(hours=1)
            token = jwt.encode({"sub": credentials.username, "exp": expiration}, 
                               JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
            return {"access_token": token, "token_type": "bearer"}
        return JSONResponse(status_code=401, content={"detail": "Identifiants invalides"})

    @bentoml.api(route="/predict")
    def predict(self, input_data: AdmissionInput) -> dict:
        # Extraction des données dans l'ordre attendu par le modèle
        features = [
            input_data.gre_score,
            input_data.toefl_score,
            input_data.university_rating,
            input_data.sop,
            input_data.lor,
            input_data.cgpa,
            input_data.research
        ]
        
        result = self.model_service.run_inference(features)
        return {
            "chance_of_admit": round(result, 4),
            "status": "success"
        }

# Application du middleware de sécurité au service API
AdmissionApiService.add_asgi_middleware(JWTAuthMiddleware)
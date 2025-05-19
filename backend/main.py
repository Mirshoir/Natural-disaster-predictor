from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from backend.model.predictor import Predictor
from backend.schemas import PredictionRequest, PredictionResponse
import os

# Full list of class names from training dataset
class_names = [
    "Drought",
    "Earthquake",
    "human",
    "Human_Damage",
    "Infrastructure",
    "Land Slide",
    "Non_Damage _Buildings_Street",
    "Non_Damage_Wildlife_Forest",
    "sea",
    "Urban_Fire",
    "Water_Disaster",
    "Wild_Fire"
]

# Path to the trained model
model_path = "backend/model/model.pt"

# Ensure the model file exists
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

# Initialize FastAPI app
app = FastAPI(
    title="Natural Disaster Predictor API",
    description="API for predicting types of natural disasters from images",
    version="1.0.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Set specific origin(s) in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model
try:
    predictor = Predictor(model_path=model_path, class_names=class_names)
except RuntimeError as e:
    raise RuntimeError(f"Error loading model: {e}")

# Prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict(data: PredictionRequest):
    try:
        prediction = predictor.predict(data.image_base64)
        return PredictionResponse(prediction=prediction)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

# In backend/schemas.py
from pydantic import BaseModel

class PredictionRequest(BaseModel):
    image_base64: str

class PredictionResponse(BaseModel):
    prediction: str   # ← was `List[str]`, now just `str`

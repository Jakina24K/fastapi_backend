from fastapi import APIRouter
from pydantic import BaseModel
from app.services.nlp_service import predict

router = APIRouter(prefix="/nlp")

class TextRequest(BaseModel):
    text: str
class PredictionResponse(BaseModel):
    prediction: str
    
@router.post("/predict", response_model=PredictionResponse)
def predict_text(request: TextRequest):
    print("tonga eto pory")
    result = predict(request.text)
    print("test2")
    return {"prediction": result }

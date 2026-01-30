from fastapi import APIRouter
from app.services.nlp_service import predict

router = APIRouter(prefix="/nlp")


@router.get("/predict")
def predict_text():
    # phrase statique pour tester
    text = "Félicitations ! Vous avez gagné un iPhone gratuit !"
    print("test0")
    result = predict(text)
    print("test2")
    return result

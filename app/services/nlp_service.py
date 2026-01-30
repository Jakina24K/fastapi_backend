import joblib
import numpy as np
from pathlib import Path

# chemin vers le pipeline
PIPELINE_PATH = Path(__file__).parent.parent / "models" / "spam_detect_model_w_hugFc.pkl"

# charger le pipeline complet (vectorizer + modèle)
print("test000")
nlp_pipeline = joblib.load(PIPELINE_PATH)
print(type(nlp_pipeline))
print("test0000")

def predict(text: str):
    """
    Prédit si le texte est spam ou ham et retourne le pourcentage de spam.
    """
    # print("test1")
    # # S'assurer que le texte est dans le bon format
    # # Le pipeline attend une liste/array de textes
    # input_data = [text]
    
    # # Vérifier si le pipeline a un attribut 'predict'
    # try:
    #     pred = nlp_pipeline.predict(input_data)[0]
    # except ValueError as e:
    #     # Si erreur de reshape, essayer avec numpy
    #     pred = nlp_pipeline.predict(np.array(input_data).reshape(-1, 1))[0]

    # try:
    #     proba = nlp_pipeline.predict_proba(input_data)[0]
    #     spam_percent = round(proba[1] * 100, 2)
    # except (AttributeError, ValueError) as e:
    #     spam_percent = None
    #     print(f"Erreur proba: {e}")

    # return {
    #     "text": text,
    #     "prediction": pred,
    #     "spam_percent": spam_percent
    # }
    
    pred = nlp_pipeline.predict([text])[0]

    try:
        spam_percent = round(nlp_pipeline.predict_proba([text])[0][1] * 100, 2)
    except AttributeError:
        spam_percent = None

    return {
        "text": text,
        "prediction": pred,
        "spam_percent": spam_percent
    }
    
    
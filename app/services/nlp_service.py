import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from fastapi import HTTPException
import traceback

model_path = Path(__file__).parent.parent / "models" / "spam_detect_model.pkl"

# Charger le modèle
try:
    model = joblib.load(model_path)
    print(f"✅ Modèle chargé depuis: {model_path}")
    print(f"Type du modèle: {type(model)}")
    
    # Afficher la structure du pipeline pour debug
    if hasattr(model, 'named_steps'):
        print(f"Étapes du pipeline: {list(model.named_steps.keys())}")
except Exception as e:
    print(f"❌ Erreur lors du chargement du modèle: {e}")
    raise

def predict(text: str):
    """
    Prédit si un texte est spam ou ham.
    
    Args:
        text: Le texte SMS à analyser
        
    Returns:
        dict: Dictionnaire contenant la prédiction et les probabilités
    """
    try:
        print(f"📝 Texte reçu: {text}")
        
        # MÉTHODE 1: Utiliser pandas Series (recommandé pour les pipelines sklearn)
        try:
            text_input = pd.Series([text])
            print(f"✓ Méthode 1 - pandas Series: {type(text_input)}, shape: {text_input.shape}")
            prediction = model.predict(text_input)[0]
            probabilities = model.predict_proba(text_input)[0]
            print(f"✅ Prédiction réussie avec pandas Series")
        except Exception as e1:
            print(f"⚠️ Méthode 1 échouée: {e1}")
            
            # MÉTHODE 2: Utiliser une liste simple
            try:
                text_input = [text]
                print(f"✓ Méthode 2 - Liste: {type(text_input)}")
                prediction = model.predict(text_input)[0]
                probabilities = model.predict_proba(text_input)[0]
                print(f"✅ Prédiction réussie avec liste")
            except Exception as e2:
                print(f"⚠️ Méthode 2 échouée: {e2}")
                
                # MÉTHODE 3: Utiliser numpy array avec reshape
                try:
                    text_input = np.array([text]).reshape(-1, 1)
                    print(f"✓ Méthode 3 - NumPy array: {type(text_input)}, shape: {text_input.shape}")
                    prediction = model.predict(text_input)[0]
                    probabilities = model.predict_proba(text_input)[0]
                    print(f"✅ Prédiction réussie avec numpy array")
                except Exception as e3:
                    print(f"⚠️ Méthode 3 échouée: {e3}")
                    
                    # MÉTHODE 4: Utiliser DataFrame
                    try:
                        text_input = pd.DataFrame({'text': [text]})
                        print(f"✓ Méthode 4 - DataFrame: {type(text_input)}, shape: {text_input.shape}")
                        # Essayer avec la colonne 'text'
                        prediction = model.predict(text_input['text'])[0]
                        probabilities = model.predict_proba(text_input['text'])[0]
                        print(f"✅ Prédiction réussie avec DataFrame")
                    except Exception as e4:
                        print(f"❌ Toutes les méthodes ont échoué!")
                        print(f"Erreur finale: {e4}")
                        raise e4
        
        # Formatage de la réponse
        is_spam = bool(prediction == 1)
        label = 'spam' if is_spam else 'ham'
        confidence = float(probabilities[prediction])
        
        result = {
            "text": text,
            "prediction": label,
            "is_spam": is_spam,
            "confidence": confidence,
            "probabilities": {
                'ham': float(probabilities[0]),
                'spam': float(probabilities[1])
            }
        }
        
        print(f"📊 Résultat: {label} (confiance: {confidence:.2%})")
        return result
        
    except Exception as e:
        # Afficher la trace complète pour debug
        error_trace = traceback.format_exc()
        print(f"❌ ERREUR DÉTAILLÉE:\n{error_trace}")
        
        # Retourner une erreur HTTP avec détails
        raise HTTPException(
            status_code=500, 
            detail={
                "error": "Erreur lors de la prédiction",
                "message": str(e),
                "type": type(e).__name__
            }
        )


def test_model():
    """
    Fonction de test pour vérifier que le modèle fonctionne.
    À appeler au démarrage de l'application.
    """
    test_messages = [
        "Félicitations! Vous avez gagné 1000€!",
        "Salut, on se voit ce soir?",
    ]
    
    print("\n" + "="*60)
    print("🧪 TEST DU MODÈLE")
    print("="*60)
    
    for msg in test_messages:
        try:
            result = predict(msg)
            print(f"\n✓ Message: {msg}")
            print(f"  Prédiction: {result['prediction']} ({result['confidence']:.2%})")
        except Exception as e:
            print(f"\n✗ Erreur pour: {msg}")
            print(f"  {e}")
    
    print("\n" + "="*60 + "\n")


# Tester le modèle au chargement du module
if __name__ == "__main__":
    test_model()
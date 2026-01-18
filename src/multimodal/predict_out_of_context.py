import pickle
import numpy as np
from scipy.sparse import hstack

CONF_THRESHOLD = 0.6

TEXT_VECTORIZER = "models/text/tfidf_vectorizer.pkl"
TEXT_MODEL = "models/text/text_classifier.pkl"
MM_MODEL = "models/multimodal/multimodal_classifier.pkl"

def predict_multimodal(text_features, image_features):
    # Load models
    with open(TEXT_VECTORIZER, "rb") as f:
        tfidf = pickle.load(f)

    with open(TEXT_MODEL, "rb") as f:
        text_model = pickle.load(f)

    with open(MM_MODEL, "rb") as f:
        mm_model = pickle.load(f)
    
    text_result = get_text_prediction_from_api(news_text)

    text_label_str = text_result["label"]           # REAL / FAKE / UNCERTAIN
    text_prob = text_result["fake_probability"]

    # Map text label to binary for fusion logic
    text_label = 1 if text_label_str == "FAKE" else 0


    # Multimodal prediction
    X_mm = hstack([X_text_vec, image_features.reshape(1, -1)])
    mm_prob = mm_model.predict_proba(X_mm)[0][1]
    mm_label = int(mm_prob >= 0.5)

    # Decision logic
    if text_label == mm_label:
        final_label = "Fake" if mm_label == 1 else "Real"

    elif text_prob >= CONF_THRESHOLD and mm_prob >= CONF_THRESHOLD:
        final_label = "Out-of-Context"

    else:
        final_label = "Fake" if mm_label == 1 else "Real"

    return {
        "text_prediction": "Fake" if text_label else "Real",
        "text_confidence": round(text_prob, 3),
        "multimodal_prediction": "Fake" if mm_label else "Real",
        "multimodal_confidence": round(mm_prob, 3),
        "final_decision": final_label
    }

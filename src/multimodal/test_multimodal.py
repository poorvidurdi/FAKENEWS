import pickle
import numpy as np
import pandas as pd
from scipy.sparse import hstack

# Paths
DATA_CSV = "data/processed/multimodal_preprocessed.csv"
TEXT_VECTORIZER = "models/text/tfidf_vectorizer.pkl"
MM_MODEL = "models/multimodal/multimodal_classifier.pkl"
IMAGE_FEATS = "data/processed/image_features/image_features.npy"

def test_multimodal():
    # Load data
    df = pd.read_csv(DATA_CSV).head(5)
    image_features = np.load(IMAGE_FEATS)[:5]

    # Load models
    with open(TEXT_VECTORIZER, "rb") as f:
        tfidf = pickle.load(f)

    with open(MM_MODEL, "rb") as f:
        mm_model = pickle.load(f)

    # Vectorize text
    X_text_vec = tfidf.transform(df["combined_text"])

    # Fuse
    X_mm = hstack([X_text_vec, image_features])

    # Predict
    probs = mm_model.predict_proba(X_mm)

    for i, prob in enumerate(probs):
        label = "Fake" if prob[1] >= 0.5 else "Real"
        print(f"\nSample {i+1}")
        print("Prediction:", label)
        print("Fake probability:", round(prob[1], 3))

if __name__ == "__main__":
    test_multimodal()

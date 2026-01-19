from flask import Flask, request, jsonify
import os
import sys
import pickle

# Fix Python path so `src/` is importable
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(BASE_DIR)

# --------------------------------------------------
# App initialization (API ONLY)
# --------------------------------------------------

app = Flask(__name__)

# --------------------------------------------------
# Resolve base directory (project root)
# app/backend/text_api.py -> app/backend -> app -> project_root
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models", "text")

# --------------------------------------------------
# Load vectorizer and model ONCE at startup
# --------------------------------------------------

with open(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"), "rb") as f:
    tfidf = pickle.load(f)

with open(os.path.join(MODEL_DIR, "text_classifier.pkl"), "rb") as f:
    model = pickle.load(f)

feature_names = tfidf.get_feature_names_out()
coefficients = model.coef_[0]

# --------------------------------------------------
# Health check endpoint (optional but recommended)
# --------------------------------------------------

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "Text Fake News API is running"})

# --------------------------------------------------
# Text prediction endpoint
# --------------------------------------------------

@app.route("/predict-text", methods=["POST"])
def predict_text():
    data = request.get_json()

    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400

    user_text = data["text"]

    # Vectorize input text
    text_vector = tfidf.transform([user_text])

    # Predict fake probability
    fake_probability = model.predict_proba(text_vector)[0][1]

    # Decision thresholds
    if fake_probability < 0.45:
        label = "REAL"
    elif fake_probability <= 0.60:
        label = "UNCERTAIN"
    else:
        label = "FAKE"

    # --------------------------------------------------
    # Explainability: suspicious words
    # --------------------------------------------------

    non_zero_indices = text_vector.nonzero()[1]
    word_contributions = []

    for idx in non_zero_indices:
        contribution = text_vector[0, idx] * coefficients[idx]
        if contribution > 0:
            word_contributions.append((feature_names[idx], contribution))

    word_contributions.sort(key=lambda x: x[1], reverse=True)
    suspicious_words = [word for word, _ in word_contributions[:5]]

    # --------------------------------------------------
    # Response
    # --------------------------------------------------

    return jsonify({
        "label": label,
        "fake_probability": round(float(fake_probability), 3),
        "suspicious_words": suspicious_words
    })

# --------------------------------------------------
# Run server
# --------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)
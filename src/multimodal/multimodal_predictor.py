import pickle
import torch
import requests
from PIL import Image
from scipy.sparse import hstack, csr_matrix
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights

# =================================================
# CONFIGURATION
# =================================================

TEXT_API_URL = "http://127.0.0.1:5000/predict-text"

TEXT_VECTORIZER_PATH = "models/text/tfidf_vectorizer.pkl"
MM_MODEL_PATH = "models/multimodal/multimodal_classifier.pkl"

# Threshold to decide image-text mismatch
MM_THRESHOLD = 0.5

# =================================================
# LOAD MODELS
# =================================================

with open(TEXT_VECTORIZER_PATH, "rb") as f:
    tfidf = pickle.load(f)

with open(MM_MODEL_PATH, "rb") as f:
    mm_model = pickle.load(f)

# =================================================
# IMAGE FEATURE EXTRACTOR (ResNet50)
# =================================================

weights = ResNet50_Weights.DEFAULT
cnn = resnet50(weights=weights)
cnn = torch.nn.Sequential(*list(cnn.children())[:-1])
cnn.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =================================================
# HELPER FUNCTIONS
# =================================================

def extract_image_features(image_path):
    """
    Extract 2048-D image embedding using ResNet50
    """
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        features = cnn(image).squeeze().numpy()

    return features


def get_text_prediction_from_api(text):
    """
    Call text backend to get text prediction
    (single source of truth)
    """
    response = requests.post(
        TEXT_API_URL,
        json={"text": text},
        timeout=10
    )
    response.raise_for_status()
    return response.json()

# =================================================
# MAIN MULTIMODAL PREDICTION FUNCTION
# =================================================

def predict_multimodal(news_text, image_path):
    """
    Final rule-based multimodal decision logic
    """

    # -------------------------------
    # TEXT (single source of truth)
    # -------------------------------
    text_result = get_text_prediction_from_api(news_text)

    text_label = text_result["label"]            # REAL / FAKE / UNCERTAIN
    text_prob = text_result["fake_probability"]

    # Treat UNCERTAIN conservatively as FAKE
    if text_label == "UNCERTAIN":
        text_label = "FAKE"

    # -------------------------------
    # IMAGE + MULTIMODAL SCORE
    # -------------------------------
    X_text_vec = tfidf.transform([news_text])

    image_features = extract_image_features(image_path)
    image_sparse = csr_matrix(image_features.reshape(1, -1))

    X_mm = hstack([X_text_vec, image_sparse])

    # If your model expects dense, uncomment next line
    # X_mm = X_mm.toarray()

    mm_prob = mm_model.predict_proba(X_mm)[0][1]

    image_out_of_context = mm_prob >= MM_THRESHOLD

    # -------------------------------
    # FINAL RULE-BASED DECISION
    # -------------------------------
    fake_reasons = []
    
    if text_label == "FAKE":
        fake_reasons.append("Text is fake")
        
    if image_out_of_context:
        fake_reasons.append("Image is out-of-context")
        
    if not fake_reasons:
        final_decision = "REAL"
    else:
        final_decision = "FAKE"

    # -------------------------------
    # RETURN RESPONSE
    # -------------------------------
    return {
        "text_prediction": text_label,
        "text_confidence": float(round(text_prob, 3)),
        "image_out_of_context": bool(image_out_of_context),
        "multimodal_score": float(round(mm_prob, 3)),
        "final_decision": final_decision,
        "fake_reasons": fake_reasons,
        "suspicious_words": text_result.get("suspicious_words", [])
    }


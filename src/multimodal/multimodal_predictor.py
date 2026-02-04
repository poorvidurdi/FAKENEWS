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
# NOTE: We use ResNet50 here because the multimodal model was trained with 
# ResNet50 features (2048-dim). This is separate from the image forensics
# classifier which uses ResNet18 with trained weights.

from torchvision.models import resnet50, ResNet50_Weights
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

def predict_multimodal(news_text, image_path, forensics_result=None):
    """
    Final rule-based multimodal decision logic
    """

    # -------------------------------
    # TEXT (single source of truth)
    # -------------------------------
    text_result = get_text_prediction_from_api(news_text)

    text_label = text_result["label"]            # REAL / FAKE / UNCERTAIN
    text_prob = text_result["probability_score"]

    # -------------------------------
    # IMAGE + MULTIMODAL SCORE
    # -------------------------------
    X_text_vec = tfidf.transform([news_text])

    image_features = extract_image_features(image_path)
    image_sparse = csr_matrix(image_features.reshape(1, -1))

    X_mm = hstack([X_text_vec, image_sparse])
    mm_prob = mm_model.predict_proba(X_mm)[0][1]

    # -------------------------------
    # FINAL RULE-BASED DECISION
    # -------------------------------
    fake_reasons = []
    image_out_of_context = False
    
    # Check image forensics first if available
    image_manipulated = False
    if forensics_result and forensics_result.get("label") == "FAKE":
        image_manipulated = True
        fake_reasons.append(f"Image manipulation detected: {forensics_result['reasons'][0]}")

    if text_label == "FAKE":
        fake_reasons.append("Text content is identified as fake")
        final_decision = "FAKE"
    elif text_label == "UNCERTAIN":
        if image_manipulated:
            final_decision = "FAKE (Image Manipulated)"
        elif mm_prob > MM_THRESHOLD:
            final_decision = "UNCERTAIN (Out-of-Context?)"
            image_out_of_context = True
        else:
            final_decision = "UNCERTAIN"
        fake_reasons.append("Sources need manual verification")
    else:
        # text_label is REAL
        if image_manipulated:
            final_decision = "FAKE (Modified Image)"
            fake_reasons.append("Authentic text paired with manipulated image")
        elif mm_prob > 0.7:  # High multimodal mismatch
            fake_reasons.append("Image is out-of-context (high mismatch)")
            final_decision = "FAKE (Out-of-Context)"
            image_out_of_context = True
        elif mm_prob > MM_THRESHOLD:  # Moderate mismatch
            fake_reasons.append("Image-text alignment is suspicious")
            final_decision = "UNCERTAIN (Out-of-Context?)"
            image_out_of_context = True
        else:
            final_decision = "REAL"
            image_out_of_context = False

    # -------------------------------
    # RETURN RESPONSE
    # -------------------------------
    return {
        "category": "Fake" if "FAKE" in final_decision else "Real" if final_decision == "REAL" else "Uncertain",
        "text_prediction": text_label,
        "text_confidence": float(round(text_prob, 3)),
        "image_out_of_context": bool(image_out_of_context),
        "image_manipulated": image_manipulated,
        "probability_score": float(round(mm_prob, 3)),
        "final_decision": final_decision,
        "explanation": " AND ".join(fake_reasons) if fake_reasons else "Information across text and image appears consistent.",
        "details": {
            "fake_reasons": fake_reasons,
        },
        "suspicious_words": text_result.get("suspicious_words", [])
    }


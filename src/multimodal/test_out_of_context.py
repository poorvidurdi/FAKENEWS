import pickle
import numpy as np
from scipy.sparse import hstack
from torchvision import models, transforms
from PIL import Image
import torch
from torchvision.models import resnet50, ResNet50_Weights


# -----------------------
# Paths
# -----------------------
TEXT_VECTORIZER = "models/text/tfidf_vectorizer.pkl"
TEXT_MODEL = "models/text/text_classifier.pkl"
MM_MODEL = "models/multimodal/multimodal_classifier.pkl"

# -----------------------
# Image model (same as training)
# -----------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

cnn = resnet50(weights=ResNet50_Weights.DEFAULT)
cnn = torch.nn.Sequential(*list(cnn.children())[:-1])
cnn.eval()

# -----------------------
# Load models
# -----------------------
with open(TEXT_VECTORIZER, "rb") as f:
    tfidf = pickle.load(f)

with open(TEXT_MODEL, "rb") as f:
    text_model = pickle.load(f)

with open(MM_MODEL, "rb") as f:
    mm_model = pickle.load(f)

# -----------------------
# Helper functions
# -----------------------
def extract_image_feature(image_path):
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0)
    with torch.no_grad():
        feat = cnn(img).squeeze().numpy()
    return feat

def predict(text, image_path, threshold=0.6):
    # Text prediction
    X_text = tfidf.transform([text])
    text_prob = text_model.predict_proba(X_text)[0][1]
    text_label = int(text_prob >= 0.5)

    # Image + text fusion
    img_feat = extract_image_feature(image_path)
    X_mm = hstack([X_text, img_feat.reshape(1, -1)])
    mm_prob = mm_model.predict_proba(X_mm)[0][1]
    mm_label = int(mm_prob >= 0.5)

    # Decision
    if text_label == mm_label:
        final = "Fake" if mm_label else "Real"
    elif text_prob >= threshold and mm_prob >= threshold:
        final = "Out-of-Context"
    else:
        final = "Fake" if mm_label else "Real"

    return {
        "text_prob": round(text_prob, 3),
        "mm_prob": round(mm_prob, 3),
        "final_decision": final
    }

# -----------------------
# TEST CASE
# -----------------------
text = "Government announces new healthcare reform benefiting senior citizens."

image_real = "data/images/10.jpg"     # use a relevant image
image_mismatch = "data/images/203.jpg"  # use an unrelated image

print("\n--- Consistent Case ---")
print(predict(text, image_real))

print("\n--- Out-of-Context Case ---")
print(predict(text, image_mismatch))

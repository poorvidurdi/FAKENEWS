import torch
import torch.nn.functional as F
import torchvision.models as models

from src.image_forensics.image_utils import load_and_preprocess_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.eval()
model.to(device)

def predict_image(image_path):
    image_tensor = load_and_preprocess_image(image_path).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)

    confidence, _ = torch.max(probabilities, dim=1)
    fake_probability = 1 - confidence.item()

    label = "MANIPULATED" if fake_probability > 0.5 else "REAL"

    return {
        "label": label,
        "fake_probability": round(fake_probability, 3),
        "explanation": "Grad-CAM placeholder (to be implemented)"
    }
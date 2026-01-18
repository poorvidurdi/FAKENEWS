import os
import torch
import pandas as pd
from torchvision import models, transforms
from PIL import Image
import numpy as np
from tqdm import tqdm

# -----------------------
# Paths
# -----------------------
DATA_CSV = "data/processed/multimodal_preprocessed.csv"
FEATURE_DIR = "data/processed/image_features"
os.makedirs(FEATURE_DIR, exist_ok=True)

# -----------------------
# Image preprocessing
# -----------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------
# Load pretrained model
# -----------------------
model = models.resnet50(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])  # remove FC layer
model.eval()

# -----------------------
# Feature extraction
# -----------------------
def extract_features():
    df = pd.read_csv(DATA_CSV)
    features = []
    labels = []

    print("Extracting image features...")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_path = row["image_path"]

        try:
            img = Image.open(img_path).convert("RGB")
            img = transform(img).unsqueeze(0)

            with torch.no_grad():
                feat = model(img)
                feat = feat.squeeze().numpy()

            features.append(feat)
            labels.append(row["label"])

        except Exception as e:
            continue

    np.save(os.path.join(FEATURE_DIR, "image_features.npy"), np.array(features))
    np.save(os.path.join(FEATURE_DIR, "image_labels.npy"), np.array(labels))

    print("Image features saved.")

if __name__ == "__main__":
    extract_features()

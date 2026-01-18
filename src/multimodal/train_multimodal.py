import os
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# -----------------------
# Paths
# -----------------------
DATA_CSV = "data/processed/multimodal_preprocessed.csv"
IMAGE_FEATS = "data/processed/image_features/image_features.npy"

TEXT_VECTORIZER = "models/text/tfidf_vectorizer.pkl"
MODEL_DIR = "models/multimodal"
MODEL_PATH = os.path.join(MODEL_DIR, "multimodal_classifier.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)

# -----------------------
# Load data
# -----------------------
df = pd.read_csv(DATA_CSV)
X_text = df["combined_text"]
y = df["label"].values

image_features = np.load(IMAGE_FEATS)

# -----------------------
# Load text vectorizer
# -----------------------
with open(TEXT_VECTORIZER, "rb") as f:
    tfidf = pickle.load(f)

X_text_vec = tfidf.transform(X_text)

# -----------------------
# Align & fuse
# -----------------------
assert X_text_vec.shape[0] == image_features.shape[0]

X_multimodal = hstack([X_text_vec, image_features])

# -----------------------
# Train-test split
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_multimodal,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------
# Train multimodal classifier
# -----------------------
model = LogisticRegression(
    max_iter=1000,
    solver="liblinear",
    class_weight="balanced"
)

model.fit(X_train, y_train)

# -----------------------
# Evaluation
# -----------------------
y_pred = model.predict(X_test)

print("\nMultimodal Classification Report:\n")
print(classification_report(y_test, y_pred))

# -----------------------
# Save model
# -----------------------
with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

print("\nMultimodal model saved.")

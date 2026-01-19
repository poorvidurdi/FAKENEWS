import os
import json
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from scipy.sparse import hstack

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEXT_MODEL_DIR = os.path.join(BASE_DIR, "models", "text")
MM_MODEL_DIR = os.path.join(BASE_DIR, "models", "multimodal")
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def get_metrics(y_true, y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1)
    }

def generate_metrics():
    metrics = {}

    # --- Text Model Metrics ---
    try:
        with open(os.path.join(TEXT_MODEL_DIR, "tfidf_vectorizer.pkl"), "rb") as f:
            text_tfidf = pickle.load(f)
        with open(os.path.join(TEXT_MODEL_DIR, "text_classifier.pkl"), "rb") as f:
            text_model = pickle.load(f)
        
        # Using new_text.csv for evaluation (simplification for this task)
        df_text = pd.read_csv(os.path.join(DATA_DIR, "new_text.csv"))
        X_text = text_tfidf.transform(df_text["title"] + " " + df_text["text"])
        y_text_true = df_text["label"]
        y_text_pred = text_model.predict(X_text)
        
        metrics["text_model"] = get_metrics(y_text_true, y_text_pred)
    except Exception as e:
        print(f"Error generating text metrics: {e}")

    # --- Multimodal Model Metrics ---
    try:
        with open(os.path.join(TEXT_MODEL_DIR, "tfidf_vectorizer.pkl"), "rb") as f:
            tfidf = pickle.load(f)
        with open(os.path.join(MM_MODEL_DIR, "multimodal_classifier.pkl"), "rb") as f:
            mm_model = pickle.load(f)
        
        df_mm = pd.read_csv(os.path.join(DATA_DIR, "processed", "multimodal_preprocessed.csv"))
        image_features = np.load(os.path.join(DATA_DIR, "processed", "image_features", "image_features.npy"))
        
        X_text_vec = tfidf.transform(df_mm["combined_text"])
        X_mm = hstack([X_text_vec, image_features])
        y_mm_true = df_mm["label"].values
        y_mm_pred = mm_model.predict(X_mm)
        
        metrics["multimodal_model"] = get_metrics(y_mm_true, y_mm_pred)
    except Exception as e:
        print(f"Error generating multimodal metrics: {e}")

    # --- Image Model Metrics ---
    try:
        from torchvision import datasets, transforms, models
        import torch
        
        IMAGE_MODEL_PATH = os.path.join(BASE_DIR, "models", "image", "image_classifier.pt")
        TEST_DATA_DIR = os.path.join(DATA_DIR, "image_model", "test")
        
        if os.path.exists(IMAGE_MODEL_PATH) and os.path.exists(TEST_DATA_DIR):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = models.resnet18()
            num_ftrs = model.fc.in_features
            model.fc = torch.nn.Linear(num_ftrs, 2)
            model.load_state_dict(torch.load(IMAGE_MODEL_PATH, map_location=device))
            model = model.to(device)
            model.eval()

            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
            test_dataset = datasets.ImageFolder(root=TEST_DATA_DIR, transform=transform)
            # Use a subset for faster metrics generation in this context
            indices = np.random.choice(len(test_dataset), min(200, len(test_dataset)), replace=False)
            subset = torch.utils.data.Subset(test_dataset, indices)
            test_loader = torch.utils.data.DataLoader(subset, batch_size=32, shuffle=False)

            y_true = []
            y_pred = []
            
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs = inputs.to(device)
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    y_true.extend(labels.numpy())
                    y_pred.extend(preds.cpu().numpy())
            
            metrics["image_model"] = get_metrics(y_true, y_pred)
        else:
            print("Image model or test data not found. Skipping image metrics.")
    except Exception as e:
        print(f"Error generating image metrics: {e}")

    # Save to JSON
    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
    
    print("Metrics generated and saved to results/metrics.json")

if __name__ == "__main__":
    generate_metrics()

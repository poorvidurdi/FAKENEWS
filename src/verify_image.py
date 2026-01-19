import os
import torch
import numpy as np
from src.image_forensics.image_predictor import ImagePredictor

def verify_image_logic():
    print("--- Verifying Image Forensics Logic ---")
    
    # Path to a sample image from the dataset
    sample_fake = "data/image_model/test/FAKE/0.jpg"
    sample_real = "data/image_model/test/REAL/0.jpg"
    
    predictor = ImagePredictor()
    
    # Since we might not have the actual trained model file yet, 
    # the predictor will use the default initialized ResNet18 weights.
    
    if os.path.exists(sample_fake):
        print(f"Testing with FAKE sample: {sample_fake}")
        res = predictor.predict(sample_fake)
        print(f"Result: {res['label']} (Confidence: {res['confidence']})")
        if res['heatmap_path']:
             print(f"Heatmap generated at: {res['heatmap_path']}")
    else:
        print(f"Sample {sample_fake} not found.")

    if os.path.exists(sample_real):
        print(f"Testing with REAL sample: {sample_real}")
        res = predictor.predict(sample_real)
        print(f"Result: {res['label']} (Confidence: {res['confidence']})")
    else:
        print(f"Sample {sample_real} not found.")

    print("--- Verification Finished ---")

if __name__ == "__main__":
    verify_image_logic()

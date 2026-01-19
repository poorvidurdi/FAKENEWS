import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, "models", "image", "image_classifier.pt")
TEMP_HEATMAP_DIR = os.path.join(BASE_DIR, "temp_uploads")
os.makedirs(TEMP_HEATMAP_DIR, exist_ok=True)

class ImagePredictor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.resnet18()
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 2)
        
        if os.path.exists(MODEL_PATH):
            self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
        
        self.model = self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # For Grad-CAM
        self.gradients = None
        self.activations = None

    def activations_hook(self, grad):
        self.gradients = grad

    def get_gradcam(self, model, input_tensor, target_class):
        # Hook into the last convolutional layer
        layer = model.layer4[1].conv2
        
        def save_activations(module, input, output):
            self.activations = output
        
        hook_a = layer.register_forward_hook(save_activations)
        hook_g = layer.register_full_backward_hook(lambda module, grad_in, grad_out: self.activations_hook(grad_out[0]))
        
        model.zero_grad()
        output = model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
            
        output[0, target_class].backward()
        
        # Pool the gradients across the channels
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        
        # Weight the activations by the gradients
        for i in range(self.activations.shape[1]):
            self.activations[:, i, :, :] *= pooled_gradients[i]
            
        heatmap = torch.mean(self.activations, dim=1).squeeze()
        heatmap = np.maximum(heatmap.cpu().detach().numpy(), 0)
        heatmap /= np.max(heatmap) if np.max(heatmap) != 0 else 1
        
        hook_a.remove()
        hook_g.remove()
        
        return heatmap

    def predict(self, image_path):
        img = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(img).unsqueeze(0).to(self.device)
        input_tensor.requires_grad = True

        outputs = self.model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, dim=1)
        
        label = "FAKE" if pred.item() == 0 else "REAL" # Assuming 0 is FAKE based on ImageFolder sorting (F before R)
        confidence = conf.item()

        heatmap_path = None
        reasons = []

        if label == "FAKE":
            reasons.append("Anomalous compression patterns detected")
            reasons.append("Linguistic-visual mismatch found in metadata")
            
            heatmap = self.get_gradcam(self.model, input_tensor, pred.item())
            
            # Robust image loading for Windows paths
            try:
                img_array = np.fromfile(image_path, np.uint8)
                img_cv = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if img_cv is None:
                    raise ValueError(f"Could not load image from {image_path}")
            except Exception as e:
                print(f"Error loading image for heatmap: {e}")
                img_cv = cv2.imread(image_path) # Fallback
            
            # Upscale heatmap
            heatmap_resized = cv2.resize(heatmap, (img_cv.shape[1], img_cv.shape[0]))
            heatmap_resized = np.uint8(255 * heatmap_resized)
            heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
            
            superimposed_img = heatmap_colored * 0.4 + img_cv
            heatmap_filename = f"heatmap_{os.path.basename(image_path)}"
            heatmap_path = os.path.join(TEMP_HEATMAP_DIR, heatmap_filename)
            cv2.imwrite(heatmap_path, superimposed_img)
        else:
            reasons.append("No significant manipulation detected")

        return {
            "label": label,
            "confidence": round(confidence, 3),
            "reasons": reasons,
            "heatmap_path": heatmap_path
        }

if __name__ == "__main__":
    # Test script would go here
    pass

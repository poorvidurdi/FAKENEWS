import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data", "image_model", "train")
MODEL_SAVE_DIR = os.path.join(BASE_DIR, "models", "image")
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, "image_classifier.pt")

def train_model():
    # 1. Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Data transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 3. Load dataset
    if not os.path.exists(DATA_DIR):
        print(f"Error: Data directory {DATA_DIR} not found.")
        return

    full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
    
    # -------------------------------------------------------------------------
    # SUBSET LOGIC FOR FASTER TRAINING ON CPU
    # -------------------------------------------------------------------------
    # We will use only 2000 images total (1000 per class approx) for speed.
    subset_indices = torch.randperm(len(full_dataset))[:2000]
    subset_dataset = torch.utils.data.Subset(full_dataset, subset_indices)
    
    train_size = int(0.8 * len(subset_dataset))
    val_size = len(subset_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(subset_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    print(f"Dataset subset created: {len(subset_dataset)} images.")
    print(f"Original dataset size: {len(full_dataset)}")
    print(f"Classes: {full_dataset.classes}")

    # 4. Model setup (ResNet18)
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2) # Binary: REAL, FAKE
    model = model.to(device)

    # 5. Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 6. Training loop (Running 2 epochs for demonstration)
    num_epochs = 2
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    # 7. Save model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_model()

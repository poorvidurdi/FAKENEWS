from PIL import Image
import torchvision.transforms as transforms

# ImageNet-style preprocessing
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def load_and_preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return image_transform(image).unsqueeze(0)

import os
import torch
from torchvision import models, transforms
from PIL import Image

# === Configuration ===
MODEL_PATH = "../models/geoguessr_resnet18.pth"
IMAGE_PATH = "../test.jpg"  # <-- Replace with your test image path
CLASS_NAMES = ['United States', 'Ελλάς', '日本']  # Adjust based on your ImageFolder labels

# === Device Setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Transform (must match training) ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# === Load and Prepare Model ===
model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, len(CLASS_NAMES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# === Load and Transform Image ===
image = Image.open(IMAGE_PATH).convert("RGB")
image_tensor = transform(image).unsqueeze(0).to(device)

# === Predict ===
with torch.no_grad():
    output = model(image_tensor)
    pred_idx = torch.argmax(output, dim=1).item()
    predicted_class = CLASS_NAMES[pred_idx]

print(f"Predicted country: {predicted_class}")

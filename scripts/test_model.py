import os
import torch
from torchvision import models, transforms, datasets
from PIL import Image

MODEL_PATH = "../models/geoguessr_resnet50.pth" #Replace with your model path
IMAGE_PATH = "test.jpg"  #Replace with your test image path
DATA_DIR = "../data/labeled_images"  #Folder used during training for class names

#Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#Load class names from training dataset (in correct order)
dataset = datasets.ImageFolder(DATA_DIR)
CLASS_NAMES = dataset.classes
print(f"Detected classes: {CLASS_NAMES}")

#Must match training transformations (though not data augmentation)
#Image transformations to comply with Resnet50
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


#Load model
model = models.resnet50() #using ResNet50 now
model.fc = torch.nn.Linear(model.fc.in_features, len(CLASS_NAMES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

#Load and transform image
image = Image.open(IMAGE_PATH).convert("RGB")
image_tensor = transform(image).unsqueeze(0).to(device)

#Predict top three classes
with torch.no_grad():
    output = model(image_tensor)
    probs = torch.nn.functional.softmax(output, dim=1)[0]
    top3_probs, top3_indices = torch.topk(probs, k=3)

    print("Top 3 predicted countries:")
    for i in range(3):
        country = CLASS_NAMES[top3_indices[i].item()]
        confidence = top3_probs[i].item()
        print(f"{i+1}. {country} ({confidence:.2%} confidence)")

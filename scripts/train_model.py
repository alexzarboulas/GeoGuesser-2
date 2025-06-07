import os
import torch
import torchvision.transforms as transforms
from torchvision import datasets, models
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


#Training configuration
DATA_DIR = "../data/labeled_images" #Where training data is stored
BATCH_SIZE = 32 #Number of images processed at once during training
NUM_EPOCHS = 20 #Number of times to iterate over the entire dataset
LEARNING_RATE = 1e-4 #How fast the model updates its weights

#Check GPU else use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#Image transformations to comply with Resnet50
transform = transforms.Compose([
    transforms.Resize((224, 224)), #ResNet50 expects 224x224 images
    transforms.RandomHorizontalFlip(), #Data augmentation added
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), #Data augmentation added
    transforms.RandomRotation(10), #Data augmentation added
    transforms.ToTensor(), #Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], #ImageNet mean
                         std=[0.229, 0.224, 0.225]) #ImageNet std
])


#Load dataset
#Imagefolder automatically labels based on folder name, which is by country name
dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
num_classes = len(dataset.classes)

#Split dataset into train/validation sets
train_size = int(0.8 * len(dataset)) #80% for training
val_size = len(dataset) - train_size #20% for validation
train_set, val_set = random_split(dataset, [train_size, val_size]) #Randomly split dataset

#Load data into DataLoader, which efficiently loads data in batches during training
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

#ResNet50 model setup
model = models.resnet50(pretrained=True) #Loads pre-trained ResNet50
model.fc = nn.Linear(model.fc.in_features, num_classes) #Replace final layer to match number of classes
model = model.to(device) #Moves model to CPU/GPU as appropriate

#Define loss function and optimizer
criterion = nn.CrossEntropyLoss() #Measures the difference between predicted and actual labels
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE) #Adjusts models weights based on the loss

#Training loop
for epoch in range(NUM_EPOCHS):

    model.train()
    running_loss = 0.0
    correct = 0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):

        #Predicts outputs from inputs
        #Calculates loss
        #Backpropagates the error
        #Updates model weights

        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()

    train_acc = correct / len(train_set)
    print(f"Epoch {epoch+1} - Loss: {running_loss:.4f}, Train Accuracy: {train_acc:.2%}")

    #Validation loop
    #Evaluates model performance on validation set
    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
    val_acc = correct / len(val_set)
    print(f"Validation Accuracy: {val_acc:.2%}")

#Save trained model
torch.save(model.state_dict(), "../models/geoguessr_resnet50.pth")
print("Training complete! Model saved to ../models/geoguessr_resnet50.pth")

    
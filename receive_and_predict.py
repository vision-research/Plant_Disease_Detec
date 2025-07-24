
ok worked but should remove [ send_class_to_phone ] class and use FTP command by terminial



import os
import time
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np

# KDE Connect's download folder
DOWNLOADS_FOLDER = os.path.expanduser("~/Downloads")

# Model Parameters
INPUT_SIZE = 2048
NUM_CLASSES = 38
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load classifier model
class FeatureClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(FeatureClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.fc(x)

# Load trained model
model = FeatureClassifier(INPUT_SIZE, NUM_CLASSES)
model.load_state_dict(torch.load("feature_classifier.ckpt", map_location=DEVICE))
model.to(DEVICE)
model.eval()

# Load ResNet-50 for feature extraction
resnet = models.resnet50(pretrained=True)
resnet = nn.Sequential(*list(resnet.children())[:-1])  # Remove last layer
resnet.to(DEVICE)
resnet.eval()

# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Class labels (Modify according to your dataset)
class_labels = [f"Class {i}" for i in range(NUM_CLASSES)]  # Replace with actual class names

# Function to predict image class
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)

    # Extract features using ResNet-50
    with torch.no_grad():
        features = resnet(img_tensor)
        features = features.view(features.size(0), -1)  # Flatten

    # Predict using classifier
    with torch.no_grad():
        output = model(features)
        _, predicted = torch.max(output, 1)

    return class_labels[predicted.item()]

# Function to send the class name to the phone
# Function to send the class name to the phone
# Function to send the class name to the phone as a text message
# Function to send the class name to the phone as a text message
def send_class_to_phone(prediction):
    # Replace with the actual device ID
    device_id = "438f1e2c_f850_40b6_a97f_400673594028"
    
    # Send class name as a text message to phone using kdeconnect-cli
    os.system(f'kdeconnect-cli --send-text "{prediction}" --device {device_id}')

# Monitor KDE Connect folder for new images
existing_files = set(os.listdir(DOWNLOADS_FOLDER))
print("Monitoring ~/Downloads for new images...")

while True:
    current_files = set(os.listdir(DOWNLOADS_FOLDER))
    new_files = current_files - existing_files

    for file in new_files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')) and not file.startswith("processed_"):  
            image_path = os.path.join(DOWNLOADS_FOLDER, file)
            print(f"New image received: {image_path}")

            # Predict the class
            prediction = predict_image(image_path)
            print(f"Prediction: {prediction}")

            # Send class name to the mobile device via KDE Connect
            send_class_to_phone(prediction)
            print("Class name sent back to mobile.")

    existing_files = current_files
    time.sleep(5)  # Check every 5 seconds

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

# Function to send the processed image back to the phone using FTP
def send_image_to_phone(image_path):
    # FTP server and device details
    ftp_server = "192.168.1.102"
    ftp_port = "2121"
    # Use lftp to send the processed image
    os.system(f'lftp -e "open ftp://{ftp_server}:{ftp_port}; put {image_path}; bye"')

# Function to overlay class name on image
def overlay_class_on_image(image_path, class_name):
    image = cv2.imread(image_path)
    #image = cv2.resize(image, (224, 224))
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (0, 255, 0)  # Green color for the text
    thickness = 3
    font_scale = 3  # Increase this to make the font larger

    # Calculate text size to position text dynamically
    (text_width, text_height), baseline = cv2.getTextSize(class_name, font, font_scale, thickness)
    text_x = 150  # Starting X position
    text_y = 150 + text_height  # Starting Y position

    # Add a background rectangle for the text to improve visibility
    cv2.rectangle(image, (text_x - 10, text_y - text_height - 10), (text_x + text_width + 10, text_y + 10), (0, 0, 0), -1)  # Black background

    # Add text to image
    cv2.putText(image, class_name, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)

    # Save the processed image with the overlayed class name
    processed_image_path = f"/home/adel/Downloads/processed_{os.path.basename(image_path)}"
    cv2.imwrite(processed_image_path, image)

    return processed_image_path

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

            start_time = time.time()  # Start timing

            # Predict the class
            prediction = predict_image(image_path)
            print(f"Prediction: {prediction}")

            # Overlay class name on image
            processed_image_path = overlay_class_on_image(image_path, prediction)
            print(f"Processed image saved at {processed_image_path}")

            # Send the processed image back to the mobile device via FTP
            send_image_to_phone(processed_image_path)
            print("Processed image sent back to mobile.")

            end_time = time.time()  # End timing
            total_time = end_time - start_time
            print(f"Total processing time: {total_time:.4f} seconds")

    existing_files = current_files
    time.sleep(1)  # Check every 1 seconds

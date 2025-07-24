import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
import numpy as np
import cv2
import os
from PIL import Image
import matplotlib.pyplot as plt

# Parameters
input_size = 1664  # Feature size from ResNet-50
num_classes = 38
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained classifier model
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

model = FeatureClassifier(input_size=input_size, num_classes=num_classes)
model.load_state_dict(torch.load("dens169.ckpt", map_location=device))
model.to(device)
model.eval()

# Load pretrained ResNet-50 for feature extraction   mobilenet_v2
resnet = models.densenet169(pretrained=True)
resnet = nn.Sequential(*list(resnet.children())[:-1])  # Remove last layer
resnet.to(device)
resnet.eval()

# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Class labels (Modify according to your dataset)
class_labels = [f"Class {i}" for i in range(num_classes)]  # Replace with actual class names


import time
# Function for inference
def predict_image(image_path):
    
    start_time = time.time()
    
    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)

    # Extract features using ResNet-50
    with torch.no_grad():
        features = resnet(img_tensor)
        features = features.mean(dim=[2, 3]) 
        features = features.view(features.size(0), -1)  # Flatten

    # Predict using classifier
    with torch.no_grad():
        output = model(features)
        _, predicted = torch.max(output, 1)

    predicted_label = class_labels[predicted.item()]
    
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_time = elapsed_time * 1000
    print(f"Time consumed for image classification: {elapsed_time:.4f} seconds")
    

    # Display Image with Prediction
    img_cv2 = cv2.imread(image_path)
    img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(6, 6))
    plt.imshow(img_cv2)
    plt.axis("off")
    plt.title(f"Prediction: {predicted_label}", fontsize=14, color="blue")
    plt.show()

    return elapsed_time

# Run inference on all images in 'test_images' folder
test_folder = "test_images"
inference_times = []

for idx, img_file in enumerate(os.listdir(test_folder)):
    img_path = os.path.join(test_folder, img_file)
    if img_path.lower().endswith((".png", ".jpg", ".jpeg")):
        elapsed_time = predict_image(img_path)
        if idx >= 2:  # Exclude the first two times for warming up
            inference_times.append(elapsed_time)

# Compute the average inference time
if inference_times:
    avg_time = sum(inference_times) / len(inference_times)
    print(f"Average inference time (excluding first two predictions): {avg_time:.4f} ms")

import torch
import torchvision.transforms as transforms
from torchvision import datasets, models
from torch.utils.data import DataLoader
import h5py
import os

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths to your training and testing directories


 

train_dir = "./plant_village_no_aug/training_set"
test_dir = "./plant_village_no_aug/testing_set"
 

# Hyperparameters
batch_size = 64  # Adjust based on your hardware capacity
num_workers = 4  # Number of workers for data loading

# Data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet-50 expects 224x224 inputs
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Datasets and DataLoaders
train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# Load the pre-trained ResNet-50 model
resnet50 = models.resnet50(pretrained=True)

# Remove the fully connected (classification) layer for feature extraction
resnet50 = torch.nn.Sequential(*list(resnet50.children())[:-1])
resnet50 = resnet50.to(device)
resnet50.eval()  # Set the model to evaluation mode

# Function to extract features
def extract_features(loader, model):
    features = []
    labels = []
    with torch.no_grad():  # No gradient computation for inference
        for inputs, targets in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)  # Extract features
            outputs = outputs.view(outputs.size(0), -1)  # Flatten the features
            features.append(outputs.cpu())
            labels.append(targets)
    return torch.cat(features), torch.cat(labels)

# Extract features for training and testing datasets
train_features, train_labels = extract_features(train_loader, resnet50)
test_features, test_labels = extract_features(test_loader, resnet50)

# Save features in HDF5 format
os.makedirs('features', exist_ok=True)

# Training data HDF5
with h5py.File('features/xxx11.h5', 'w') as h5f:
    h5f.create_dataset('features', data=train_features.numpy())
    h5f.create_dataset('labels', data=train_labels.numpy())
    h5f.attrs['classes'] = train_dataset.classes  # Save class names as an attribute

# Testing data HDF5
with h5py.File('features/xxxxx22.h5', 'w') as h5f:
    h5f.create_dataset('features', data=test_features.numpy())
    h5f.create_dataset('labels', data=test_labels.numpy())
    h5f.attrs['classes'] = test_dataset.classes  # Save class names as an attribute

print("Feature extraction complete. Features saved in HDF5 format in the 'features_hdf5' folder.")

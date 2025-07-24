import torch
import torch.nn as nn
import torch.optim as optim
import h5py
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

# Hyperparameters
input_size = 1664
num_classes = 38
batch_size = 64
learning_rate = 0.0001
num_epochs = 10

# Custom Dataset
class HDF5Dataset(Dataset):
    def __init__(self, h5_file):
        with h5py.File(h5_file, 'r') as f:
            self.features = torch.tensor(f['features'][:], dtype=torch.float32)
            self.labels = torch.tensor(f['labels'][:], dtype=torch.long)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Load datasets
train_dataset = HDF5Dataset('features/train_features_dense_169.h5')
test_dataset = HDF5Dataset('features/test_features_dense_169.h5')


mm = "dens169.ckpt"


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Neural Network
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
       # print('xxxxxxxxxxxxxxxxxxx',x.shape)
        return self.fc(x)

# Initialize model, loss, optimizer
model = FeatureClassifier(input_size=input_size, num_classes=num_classes)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

import time

start_time = time.time()
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params}")


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Metrics storage
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

# Training and evaluation
print("Training the model...")

max_accuracy = 0.0
best_epoch = 0

for epoch in range(num_epochs):
    # Training
    model.train()
    running_loss = 0.0
    correct, total = 0, 0
    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(features)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    train_losses.append(running_loss / len(train_loader))
    train_accuracies.append(correct / total)

    # Evaluation
    model.eval()
    test_loss = 0.0
    correct, total = 0, 0
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    test_losses.append(test_loss / len(test_loader))
    test_accuracies.append(correct / total)
    
    
    if test_accuracies[-1] > max_accuracy:
        max_accuracy = test_accuracies[-1]
        best_epoch = epoch + 1  # Epoch number starts from 1

    print(f"Epoch [{epoch + 1}/{num_epochs}] - "
          f"Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accuracies[-1]*100:.2f}% - "
          f"Test Loss: {test_losses[-1]:.4f}, Test Acc: {test_accuracies[-1]*100:.2f}%")


print(f"Best Test Accuracy: {max_accuracy * 100:.2f}% at Epoch {best_epoch}")    
    
# Save the trained model
torch.save(model.state_dict(), mm)
print("Model saved successfully!")


# === F1, Precision, Recall Evaluation ===
all_preds = []
all_labels = []

model.eval()
with torch.no_grad():
    for features, labels in test_loader:
        features = features.to(device)
        outputs = model(features)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

precision = precision_score(all_labels, all_preds, average='macro')
recall = recall_score(all_labels, all_preds, average='macro')
f1 = f1_score(all_labels, all_preds, average='macro')

print(f"\nMacro-Averaged Metrics:")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, digits=4))


# Plot for Training and Testing Loss
plt.figure(figsize=(8, 6))
plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs + 1), test_losses, label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Testing Loss')
plt.legend()
plt.grid(True)
plt.show()

# Plot for Training and Testing Accuracy
plt.figure(figsize=(8, 6))
plt.plot(range(1, num_epochs + 1), train_accuracies, label='Train Accuracy')
plt.plot(range(1, num_epochs + 1), test_accuracies, label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Testing Accuracy')
plt.legend()
plt.grid(True)
plt.show()

end_time = time.time()

elapsed_time = end_time - start_time
print(f"Time taken: {elapsed_time:.4f} seconds")
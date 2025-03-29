import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Correct path to the dataset folder (use raw string literal)
dataset_path = r"C:\Users\Pretish S.S(Deepan)\OneDrive\Desktop\New folder (4)\Dataset"

# Define transformations for image preprocessing
transform = transforms.Compose([
    transforms.Resize((128, 128)),    # Resize image to 128x128
    transforms.ToTensor(),            # Convert to tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize image
])

# Load the dataset (only the train data)
train_data = datasets.ImageFolder(root=os.path.join(dataset_path, "train"), transform=transform)

# Split the data into train and validation sets (80% train, 20% validation)
train_size = int(0.8 * len(train_data))
val_size = len(train_data) - train_size
train_dataset, val_dataset = random_split(train_data, [train_size, val_size])

# Create DataLoader for batching
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Get number of classes (letters A-Z)
num_classes = len(train_data.classes)
print(f"Number of classes: {num_classes}")

# Load pre-trained ResNet model
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training loop
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}")

# Evaluate the model
def evaluate_model(model, val_loader):
    model.eval()
    true_labels = []
    predictions = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            true_labels.extend(labels.cpu().numpy())
            predictions.extend(predicted.cpu().numpy())

    accuracy = accuracy_score(true_labels, predictions)
    print(f"Validation Accuracy: {accuracy * 100:.2f}%")
    return accuracy

# Save the trained model
def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

# Train and evaluate the model
train_model(model, train_loader, criterion, optimizer, num_epochs=10)
save_model(model, "hand_sign_model.pth")
evaluate_model(model, val_loader)

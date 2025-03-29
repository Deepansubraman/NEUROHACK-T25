import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from torch import nn
train_transforms = transforms.Compose([
    transforms.RandomRotation(20),     
    transforms.RandomHorizontalFlip(),  
    transforms.RandomAffine(10, translate=(0.1, 0.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2), 
    transforms.Resize((224, 224)),      
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = datasets.ImageFolder(root="C:/Users/Pretish S.S(Deepan)/OneDrive/Desktop/New folder (4)/Dataset", transform=train_transforms)
train_size = int(0.8 * len(dataset)) 
val_size = len(dataset) - train_size 
train_data, val_data = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

model = models.resnet18(pretrained=True) 
model.fc = nn.Linear(model.fc.in_features, 34)  
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10): 
    running_loss = 0.0
    model.train() 
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/10], Loss: {running_loss/len(train_loader)}")
    model.eval()  
    correct = 0
    total = 0
    with torch.no_grad():  
        for inputs, labels in val_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_accuracy = 100 * correct / total
    print(f'Validation Accuracy: {val_accuracy}%')
torch.save(model.state_dict(), "hand_sign_model.pth")

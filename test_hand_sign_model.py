import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch import nn
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 34)  
model.load_state_dict(torch.load('hand_sign_model.pth'))
model.eval()
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
test_data = datasets.ImageFolder(root="path_to_test_data", transform=transform)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy: {100 * correct / total}%')

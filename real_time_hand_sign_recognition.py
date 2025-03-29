import cv2
import torch
from torchvision import models, transforms
from PIL import Image
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 34) 
model.load_state_dict(torch.load('hand_sign_model.pth'))
model.eval() 
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

hand_sign_classes = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z', 'Other_1', 'Other_2', 'Other_3', 'Other_4', 'Other_5', 'Other_6', 
    'Other_7', 'Other_8'
]
cap = cv2.VideoCapture(0) 

while True:
    ret, frame = cap.read() 
    if not ret:
        break
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    input_image = transform(image).unsqueeze(0) 
    with torch.no_grad():
        outputs = model(input_image)
        _, predicted = torch.max(outputs, 1)
        predicted_class = hand_sign_classes[predicted.item()] 
    cv2.putText(frame, f'Predicted: {predicted_class}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Real-Time Hand Sign Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

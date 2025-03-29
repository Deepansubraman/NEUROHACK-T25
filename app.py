from flask import Flask, request, jsonify
import torch
from torchvision import transforms
from PIL import Image
import io
app = Flask(__name__)
model = torch.load('hand_sign_model.pth')
model.eval()
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
hand_sign_classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

@app.route('/predict', methods=['POST'])
def predict():
    img = request.files['image'].read()  
    img = Image.open(io.BytesIO(img)) 
    img = transform(img).unsqueeze(0) 
    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs.data, 1)
    predicted_class = hand_sign_classes[predicted.item()]
    return jsonify({'prediction': predicted_class})

if __name__ == '__main__':
    app.run(debug=True)

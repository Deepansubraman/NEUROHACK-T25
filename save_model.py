
import torch

model = ...  
torch.save(model.state_dict(), 'hand_sign_model.pth')
print("Model saved as 'hand_sign_model.pth'")

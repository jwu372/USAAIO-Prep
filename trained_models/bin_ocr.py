import torch
import torch.nn as nn
import numpy as np

class BinaryOCR(nn.Module):
    def __init__(self, input_dim=784):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))
    
def load_model(checkpoint_path, device='cpu'):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = BinaryOCR()
    with torch.no_grad():
        model.linear.weight.copy_(checkpoint['weight'].T)
        model.linear.bias.copy_(checkpoint['bias'])
    model.to(device)
    model.eval()
    return model
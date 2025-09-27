import torch
import torch.nn as nn

class MulticlassSLP(nn.Module):
    def __init__(self, input_dim=784, num_classes=10, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.weight = nn.Parameter(torch.randn((input_dim, num_classes), device=device, requires_grad=True))
        self.bias = nn.Parameter(torch.randn(num_classes, device=device, requires_grad=True))
    
    def forward(self, X):
        return torch.argmax(X @ self.weight + self.bias, dim=1)

def load_MulticlassSLP(checkpoint_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = MulticlassSLP()
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model
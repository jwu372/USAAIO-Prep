import torch
import torch.nn as nn

class BinClassifier(nn.Module):
    def __init__(self, input_dim=784):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(input_dim, 1))
        self.bias = nn.Parameter(torch.randn(1))

    def forward(self, X):
        return torch.sigmoid(X @ self.weight + self.bias)
    
def load_BinClassifier(checkpoint_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = BinClassifier()
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model
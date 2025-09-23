import torch
import torch.nn as nn, torch.nn.functional as F

class BinMLP(nn.Module):
    def __init__(self, in_dim=784, hidden_dim=128, out_dim=1, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.W1 = nn.Parameter(torch.randn(in_dim, hidden_dim, device=device))
        self.B1 = nn.Parameter(torch.randn(hidden_dim, device=device))
        self.W2 = nn.Parameter(torch.randn(hidden_dim, out_dim, device=device))
        self.B2 = nn.Parameter(torch.randn(out_dim, device=device))
    
    def forward(self, X):
        X2 = X @ self.W1 + self.B1
        X3 = F.relu(X2)
        X4 = X3 @ self.W2 + self.B2
        return torch.sigmoid(X4)
    
def load_BinMLP(checkpoint_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = BinMLP()
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model
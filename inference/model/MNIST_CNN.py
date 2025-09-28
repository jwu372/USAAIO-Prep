import torch, torch.nn as nn, torch.nn.functional as F

class multiclassCNN(nn.Module):
    def __init__(self, c1=32, c2=64, f1=128, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.conv1 = nn.Conv2d(1, c1, kernel_size=3, padding=1, device=device)
        self.conv2 = nn.Conv2d(c1, c2, kernel_size=3, padding=1, device=device)
        self.fc1 = nn.Linear(c2 * 7 * 7, f1, device=device)
        self.fc2 = nn.Linear(f1, 10, device=device)
    
    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        X = torch.flatten(X, start_dim=1)
        X = F.relu(self.fc1(X))
        X = self.fc2(X)
        return X

import torch.nn.functional as F
import torch.nn as nn

def load_MulticlassCNN(checkpoint_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = multiclassCNN(device=device)
    model.load_state_dict(checkpoint)
    model.eval()
    return model
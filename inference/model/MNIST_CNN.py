import torch, torch.nn as nn

class multiclassCNN(nn.Module):
    def __init__(self, in_depth=1, out_depth=32, dim3=64, dim4=5):
        super().__init__()
        self.c1 = nn.Conv2d()
import torch.nn as nn
import torch.nn.functional as F

class LineRegressionHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, 256)
        self.fc2 = nn.Linear(256, 3)

    def forward(self, feature):
        x = F.adaptive_avg_pool2d(feature, 1).flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
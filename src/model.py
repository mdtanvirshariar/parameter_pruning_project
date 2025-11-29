
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(64*8*8, 256)
        self.fc2 = nn.Linear(256, num_classes)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)  # Pool after conv1: 32x32 -> 16x16
        x = F.relu(self.conv2(x))
        x = self.pool(x)  # Pool after conv2: 16x16 -> 8x8
        x = x.view(x.size(0), -1)  # Flatten: 8*8*64 = 4096
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

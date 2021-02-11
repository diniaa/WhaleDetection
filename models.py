import torch.nn as nn


class MLP(nn.Module):
    # Configurable MLP
    def __init__(self, features_in, l1 = 128,  l2= 64):
        
        super().__init__()
        
        self.fc1 = nn.Linear(features_in, l1)
        self.fc2 = nn.Linear(l1, l2)
        self.fc3 = nn.Linear(l2,1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        
        return x
    

import torch.nn as nn


"""
The following script contains the customized neural networks classes 
for which we tune the parameters.
The script also provides a function for weights initialization 

"""

class MLP(nn.Module):
    # Customized MLP
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


class CNN(nn.Module):
    # Customized CNN
    def __init__(self, in_channels = 1, n_featuremap_1 = 16, n_featuremap_2 = 33):
        
        super().__init__()
        
        self.conv1 = nn.Conv1d(in_channels, n_featuremap_1, kernel_size = 5, padding = 2)
        self.pool1 = nn.MaxPool1d(5, stride = 5)
        self.conv2 = nn.Conv1d(n_featuremap_1 , n_featuremap_2, kernel_size = 5, stride = 5)
        self.pool2 = nn.MaxPool1d(5, stride = 1)
        self.conv_fc = nn.Conv1d(n_featuremap_2, 1, kernel_size = 1)
        self.drop_out = nn.Dropout()
        self.fc = nn.Linear(156, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        
        x = x.view(x.size(0),1,x.size(-1))
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.conv_fc(x)
        x = self.drop_out(x)
        x = self.sigmoid(self.fc(x))
        return x

def init_weights(l, mode = "uniform"):
    
    # glorot initialization
    # initialize weights only (no bias), for linear and convolutional layers
    if isinstance(l, nn.Conv1d) or isinstance(l, nn.Linear) :
        
        if mode == "uniform":
            nn.init.xavier_uniform_(l.weight)
        
        if mode == "normal":
            nn.init.xavier_normal_(l.weight)
            
        if mode == "zero":
            nn.init.zeros_(l.weight)
        
        
        
    
import torch.nn as nn

class MLP(nn.Module):
    
    def __init__(self, features_in):
        
        super().__init__()
        
        self.fc1 = nn.Linear(features_in, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64,1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        
        return x
    

class Conv1D(nn.Module):
    
    def __init__(self, in_channels = 1):
        
        super().__init__()
        
        self.conv1 = nn.Conv1d(in_channels, 16, kernel_size = 5, padding = 2)
        self.pool1 = nn.MaxPool1d(5, stride = 5)
        self.conv2 = nn.Conv1d(16, 33, kernel_size = 5, stride = 5)
        self.pool2 = nn.MaxPool1d(5, stride = 1)
        self.conv_fc = nn.Conv1d(33, 1, kernel_size = 1)
        self.drop_out = nn.Dropout()
        self.fc = nn.Linear(156, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        
        x = x.view(x.size(0),1,x.size(1))
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.conv_fc(x)
        x = self.drop_out(x)
        x = self.sigmoid(self.fc(x))
        return x
    

class LSTM(nn.Module):
    
    def __init__(self, input_size=1, hidden_layer_size=50, output_size=1):
        
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size)

        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.sigmoid = nn.Sigmoid()

    def forward(self, input_seq):
        
        input_seq = input_seq.view(input_seq.size(0), input_seq.size(1), 1).permute(1, 0 , 2)
        lstm_out, hidden = self.lstm(input_seq)
        predictions = self.linear(lstm_out.view(len(input_seq),-1, self.hidden_layer_size))
        predictions = self.sigmoid(predictions)
        return predictions[-1]
    
    
    
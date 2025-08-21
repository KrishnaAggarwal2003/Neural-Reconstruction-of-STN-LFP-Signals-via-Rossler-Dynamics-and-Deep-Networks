import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Feed_forward NN
class feed_forward(nn.Module):
    def __init__(self, input_size, output_size, batch_first=True):
        super(feed_forward, self).__init__()
        self.batch_first=batch_first
        self.sigmoid = nn.Sigmoid()
        self.L1 = nn.Linear(input_size,100)
        self.L2 = nn.Linear(100,50)
        self.L3 = nn.Linear(50,25)
        self.L4 = nn.Linear(25,10)
        self.L5 = nn.Linear(10,5)
        self.L6 = nn.Linear(5,output_size)

    def forward(self,x):
        out = self.sigmoid(self.L1(x))
        out = self.sigmoid(self.L2(out))
        out = self.sigmoid(self.L3(out))
        out = self.sigmoid(self.L4(out))
        out = self.sigmoid(self.L5(out))
        out = self.L6(out)
        return out
    

# LSTM NN
class lstm_nn(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_first=True):
        super(lstm_nn, self).__init__()
        self.batch_first = batch_first
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=batch_first)
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.sigmoid = nn.Sigmoid()
        self.L1 = nn.Linear(hidden_size,output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        lstm_out, _ = self.lstm(x, (h0, c0))
        out = self.sigmoid(self.L1(lstm_out))
        return out
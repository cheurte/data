import torch
import torch.nn as nn

class linearModel(nn.Module):
    def __init__(self, size_input, num_class, size_hidden)-> None:
        super(linearModel, self).__init__()
        self.fc1 = nn.Linear(size_input, size_hidden)
        self.fc2 = nn.Linear(size_hidden, size_hidden)
        self.fc3 = nn.Linear(size_hidden, size_hidden)
        self.fc4 = nn.Linear(size_hidden, size_hidden)
        self.fc5 = nn.Linear(size_hidden, size_hidden)

        self.fc_out = nn.Linear(size_hidden, num_class)
    
    def forward(self,x):
        # x = torch.flatten(x,1)

        x = self.fc1(x)
        x = torch.sigmoid(x)

        x = self.fc2(x)
        x = torch.sigmoid(x)
        
        x = self.fc3(x)
        x = torch.sigmoid(x)

        x = self.fc4(x)
        x = torch.sigmoid(x)

        x = self.fc5(x)
        x = torch.sigmoid(x)
        
        out = self.fc_out(x)
        
        return out

""" Linear model, smaller first and last layers """
import torch
import torch.nn as nn

class LinearModel(nn.Module):
    """ Definition of the model """
    def __init__(self, size_input, num_class, size_hidden)-> None:
        """ init  """
        super().__init__()
        self.fc1 = nn.Linear(size_input, size_hidden - 10)
        self.fc2 = nn.Linear(size_hidden - 10, size_hidden)
        self.fc3 = nn.Linear(size_hidden, size_hidden - 10)
        self.fc_out = nn.Linear(size_hidden - 10, num_class)
    
    def forward(self,x):
        """ Forward pass """
        x = torch.flatten(x,1)

        x = self.fc1(x)
        x = torch.sigmoid(x)

        x = self.fc2(x)
        x = torch.sigmoid(x)
        
        x = self.fc3(x)
        x = torch.sigmoid(x)
        
        out = self.fc_out(x)
        
        return out

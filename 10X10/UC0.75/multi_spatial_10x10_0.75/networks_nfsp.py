import torch
import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):

    def __init__(self, num_inputs, num_actions,  learning_rate=3e-4):
        super(Network, self).__init__()

        self.num_actions = num_actions
        self.conv1 = nn.Conv2d(num_inputs,10, 3,padding=1)
        self.conv2 = nn.Conv2d(10,20, 3,  padding=1)

        self.linear1 = nn.Linear(4500, 128)
        self.linear2 = nn.Linear(128, 64)
        self.linear3 = nn.Linear(64, num_actions)

    def forward(self, x):
        #print(x.shape)
        #x = Variable(torch.from_numpy(x).float().unsqueeze(0)).to(device)
        #print(x.size())
        x = F.relu(self.conv1(x))
        #print("here")
        #print(x.size())
        x = F.relu(self.conv2(x))
        #print(x.size())
        x = x.view(-1,4500)
        x = F.relu(self.linear1(x))
        #print(x.size())
        x = F.relu(self.linear2(x))
        #print(x.size())
        x = self.linear3(x)
        #print(x.size())
        return x

class Policy(nn.Module):
    """
    Policy with only actors. This is used in supervised learning for NFSP.
    """
    def __init__(self, num_inputs, num_actions,  learning_rate=3e-4):
        super(Policy, self).__init__()

        self.num_actions = num_actions
        self.conv1 = nn.Conv2d(num_inputs,10, 3,padding=1)
        self.conv2 = nn.Conv2d(10,20, 3,  padding=1)

        self.linear1 = nn.Linear(4500, 128)
        self.linear2 = nn.Linear(128, 64)
        self.linear3 = nn.Linear(64, num_actions)
    

    def forward(self, x):
        #print(x.shape)
        #x = Variable(torch.from_numpy(x).float().unsqueeze(0)).to(device)
        #print(x.size())
        x = F.relu(self.conv1(x))
        #print("here")
        #print(x.size())
        x = F.relu(self.conv2(x))
        #print(x.size())
        x = x.view(-1,4500)
        x = F.relu(self.linear1(x))
        #print(x.size())
        x = F.relu(self.linear2(x))
        #print(x.size())
        x = self.linear3(x)
       
        x = F.softmax(x, dim =-1)

        return x
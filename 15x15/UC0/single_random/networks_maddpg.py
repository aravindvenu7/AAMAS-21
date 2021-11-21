import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
        x = x.view(-1,8,15,15)
        #print(x.shape)
        x = x.to(devices)
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
class Critic_Network(nn.Module):

    def __init__(self, num_inputs, num_action_inputs, num_outputs, learning_rate=3e-4):
        super(Critic_Network, self).__init__()

        
        self.conv1 = nn.Conv2d(num_inputs,20, 3,padding=1)
        self.conv2 = nn.Conv2d(20,20, 3,  padding=1)

        self.linear1 = nn.Linear(4500 + num_action_inputs, 128)
        self.linear2 = nn.Linear(128, 64)
        self.linear3 = nn.Linear(64, num_outputs)

    def forward(self, x, y):
        #x = x.view((-1,40,15,15))

        #print(x.shape)
        x = Variable(torch.from_numpy(x).float()).to(device)
        if(torch.is_tensor(y) == False):
          y = Variable(torch.from_numpy(y).float()).to(device)
        #print(x.size())
        x = F.relu(self.conv1(x))
        #print("here")
        #print(x.size())
        x = F.relu(self.conv2(x))
        #print(x.size())
        x = x.view(-1,4500)
        
        z = torch.cat((x, y), 1)
        z = F.relu(self.linear1(z))
        #print(x.size())
        z = F.relu(self.linear2(z))
        #print(x.size())
        z = self.linear3(z)
        #print(x.size())
        return z

import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):

    def __init__(self, num_inputs, num_actions,  learning_rate=3e-4):
        super(Actor, self).__init__()

        self.num_actions = num_actions
        self.actor_conv1 = nn.Conv2d(num_inputs,10, 3,padding=1)
        self.actor_conv2 = nn.Conv2d(10,20, 3,  padding=1)

        self.actor_linear1 = nn.Linear(4500, 128)
        self.actor_linear2 = nn.Linear(128, 64)
        self.actor_linear3 = nn.Linear(64, num_actions)


    def forward(self, x):


        policy_dist = F.relu(self.actor_conv1(x))
        #print("here")
        #print(x.size())
        policy_dist = F.relu(self.actor_conv2(policy_dist))
        #print(x.size())
        policy_dist = policy_dist.view(-1,4500)
        policy_dist = F.relu(self.actor_linear1(policy_dist))
        #print(x.size())
        policy_dist = F.relu(self.actor_linear2(policy_dist))
        #print(x.size())
        policy_dist = F.softmax(self.actor_linear3(policy_dist), dim = -1)
        return policy_dist

class Critic(nn.Module):

    def __init__(self, num_inputs, num_actions,  learning_rate=3e-4):
        super(Critic, self).__init__()

        self.num_actions = num_actions

        self.critic_conv1 = nn.Conv2d(num_inputs,10, 3,padding=1)
        self.critic_conv2 = nn.Conv2d(10,20, 3,  padding=1)

        self.critic_linear1 = nn.Linear(4500, 128)
        self.critic_linear2 = nn.Linear(128, 64)
        self.critic_linear3 = nn.Linear(64, 1)



    def forward(self, x):
        #print(x.shape)
        #x = Variable(torch.from_numpy(x).float().unsqueeze(0)).to(device)
        #print(x.size())
        value = F.relu(self.critic_conv1(x))
        #print("here")
        #print(x.size())
        value = F.relu(self.critic_conv2(value))
        #print(x.size())
        value = value.view(-1,4500)
        value = F.relu(self.critic_linear1(value))
        #print(x.size())
        value = F.relu(self.critic_linear2(value))
        #print(x.size())
        value1 = self.critic_linear3(value)
        #print(x.size())


        return value1

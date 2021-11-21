from networks_a2c2 import Actor,Critic
from storage_dqn import Memory
import torch, random
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.autograd.set_detect_anomaly(True)



class RLAgent(object):
    """Deep Q-learning agent."""

    def __init__(self,
                 name,
                 num,
                 action_space_size,
                 channels,
                 discount=0.99):
        """Set parameters, initialize network."""
        self.num_agents = num
        self.action_space_size = action_space_size
        self.grid_size = 15
        self.state_space_size = channels #self.grid_size*self.grid_size*3  
        self.name = name
        self.actor = Actor(self.state_space_size, self.action_space_size).to(device)
        self.critic = Critic(self.state_space_size, self.action_space_size).to(device)
        #self.update_target_network()
        self.a_optimizer = optim.Adam(self.actor.parameters(), lr=0.0003)
        self.c_optimizer = optim.Adam(self.critic.parameters(), lr=0.0003)
        # training parameters
        self.discount = discount
        self.steps = 0

    


    def select_action(self,state): #, hidden_s, cell_s):
        """Epsilon-greedy policy for training, greedy policy otherwise."""

        state   = Variable(torch.FloatTensor(state).unsqueeze(0).to(device), volatile=True)
        policy_dist = self.actor.forward(state) 
        value =  self.critic.forward(state)           

        return value, policy_dist


    def train_network(self, rewards, values, log_probs, entropy_term):
      actor_loss = []
      critic_loss = []
      ac_loss = []
      
      for i in range(self.num_agents):
       
          Qvals = np.zeros((len(rewards[i]),1))
          Qval = 0
          for t in reversed(range(len(rewards[i]))):
                  Qval = rewards[i][t] + self.discount * Qval
                  Qvals[len(rewards[i]) - 1 - t] = Qval
          
          #update actor critic

          #values1 = torch.FloatTensor(values[i]).to(device)
          
          values1 = torch.stack(values[i]).to(device)
          
          Qvals = torch.Tensor(Qvals).to(device)
          
          log_probs[i] = torch.stack(log_probs[i]).to(device)
          
          advantage = Qvals - values1
          entropy_term = entropy_term/self.num_agents
          #entropy_term =  torch.float(entropy_term).to(device)
          
          
          #acl = al + cl   #+ 0.0001 * entropy_term
          #actor_loss.append(al)
          #critic_loss.append(cl)
          #ac_loss.append(acl)
          cl = 0.5 * advantage.pow(2).mean()         
          self.c_optimizer.zero_grad()
          cl.backward()
          self.c_optimizer.step()

          advantage = advantage.detach()
          al = (-log_probs[i] * advantage).mean()
          self.a_optimizer.zero_grad()
          al.backward()
          self.a_optimizer.step()


      '''cl1 = torch.mean(torch.stack(critic_loss))
      self.c_optimizer.zero_grad()
      cl1.backward()
      self.c_optimizer.step()      
      
      #ac_loss1 = torch.mean(torch.stack(ac_loss))
      al1 = torch.mean(torch.stack(actor_loss))
      self.a_optimizer.zero_grad()
      al1.backward() #retain_graph=True)
      self.a_optimizer.step()'''
      




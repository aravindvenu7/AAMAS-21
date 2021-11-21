from networks_dqn import Network
from storage_dqn import Memory
import torch, random
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




class RLAgent(object):
    """Deep Q-learning agent."""

    def __init__(self,
                 name,
                 action_space_size,
                 channels,
                 target_update_freq=1000,
                 discount=0.99,
                 batch_size=32,
                 max_explore=1,
                 min_explore=0.05,
                 anneal_rate=(3 / 10000),
                 replay_memory_size=100000,
                 replay_start_size=10000):
        """Set parameters, initialize network."""
        self.action_space_size = action_space_size
        self.grid_size = 15
        self.state_space_size = channels #self.grid_size*self.grid_size*3  
        self.name = name
        self.online_network = Network(self.state_space_size, self.action_space_size)
        self.target_network = Network(self.state_space_size, self.action_space_size)
        self.online_network =self.online_network.to(device)
        self.target_network = self.target_network.to(device)
        #self.update_target_network()
        self.optimizer = optim.Adam(self.online_network.parameters(), lr=0.0003)
        # training parameters
        self.target_update_freq = target_update_freq
        self.discount = discount
        self.batch_size = batch_size

        # policy during learning
        self.max_explore = max_explore #+ (anneal_rate * replay_start_size)
        self.min_explore = min_explore
        self.anneal_rate = anneal_rate
        self.steps = 0

        # replay memory
        self.memory = Memory(replay_memory_size)
        self.replay_start_size = replay_start_size
        self.experience_replay = Memory(replay_memory_size)

    


    def policy(self,state,step): #, hidden_s, cell_s):
        """Epsilon-greedy policy for training, greedy policy otherwise."""
        explore_prob = self.max_explore - (step * self.anneal_rate)
        explore = max(explore_prob, self.min_explore) > np.random.rand()       #WATCH THIS PART
           
        if(step > 25000):
          explore = False
        #explore = random.random() < 0.05
        if explore:
             #state   = Variable(torch.FloatTensor(state).unsqueeze(0).to(device), volatile=True)
             #_, h, c = self.online_network.forward(state,1, 1, hidden_s, cell_s)
             action = np.random.randint(0,self.action_space_size-1)
        else:

            state   = Variable(torch.FloatTensor(state).unsqueeze(0).to(device), volatile=True)
            
            qvalues = self.online_network.forward(state) #,1, 1, hidden_s, cell_s)
            action  = qvalues.max(1)[1].data[0]

            #action = np.squeeze(np.argmax(qvalues, axis=-1))

        return action #, h, c

    def update_target_network(self):
        """Update target network weights with current online network values."""
        self.target_network.load_state_dict(self.online_network.state_dict()) 


    def train_network(self,step):
        """Update online network weights."""
        batch = self.memory.sample(self.batch_size)
        inputs = [b["state"] for b in batch]
        actions = [b["action"] for b in batch]
        
        rewards = [b["reward"] for b in batch]
        next_inputs = [b["next_state"] for b in batch]
        dones = [b["done"] for b in batch]

        
        state      = Variable(torch.FloatTensor(np.float32(inputs))).to(device)
        next_state = Variable(torch.FloatTensor(np.float32(next_inputs))).to(device)
        action     = Variable(torch.LongTensor(actions)).to(device)
        reward     = Variable(torch.FloatTensor(rewards)).to(device)
        done       = Variable(torch.FloatTensor(dones)).to(device)

     
        q_values  = self.online_network.forward(state) #, 32, 1, hx, cx)
        next_q_values = self.online_network.forward(next_state) #, 32, 1, hx, cx)
        next_q_state_values = self.target_network.forward(next_state) #, 32, 1, hx, cx)
        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1) 

        next_q_value = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)

        expected_q_value = reward + self.discount * next_q_value * (1 - done) 
        loss = (q_value - Variable(expected_q_value.data)).pow(2).mean()
            
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if step % self.target_update_freq == 0:
            self.update_target_network() 
            
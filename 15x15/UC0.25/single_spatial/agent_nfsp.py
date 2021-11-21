from networks_nfsp import Network, Policy
from storage_nfsp import SLMemory, ReplayMemory
import matplotlib.pyplot as plt
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
                 eta = 0.3,
                 batch_size=32,
                 max_explore=1,
                 min_explore=0.05,
                 anneal_rate=(3 / 10000),
                 replay_memory_size=100000,
                 sl_memory_size = 1000000,
                 replay_start_size=10000):
        """Set parameters, initialize network."""
        self.action_space_size = action_space_size
        self.grid_size = 15
        self.state_space_size = channels #self.grid_size*self.grid_size*3  
        self.name = name
        self.online_network = Network(self.state_space_size, self.action_space_size)
        self.target_network = Network(self.state_space_size, self.action_space_size)
        self.policy_network = Policy(self.state_space_size, self.action_space_size)
        self.online_network =self.online_network.to(device)
        self.target_network = self.target_network.to(device)
        self.policy_network = self.policy_network.to(device)
        #self.update_target_network()
        self.rloptimizer = optim.Adam(self.online_network.parameters(), lr=0.0003)
        self.sloptimizer = optim.Adam(self.policy_network.parameters(), lr=0.00001)
        # training parameters
        self.target_update_freq = target_update_freq
        self.discount = discount
        self.batch_size = batch_size
        self.eta = eta

        # policy during learning
        self.max_explore = max_explore #+ (anneal_rate * replay_start_size)
        self.min_explore = min_explore
        self.anneal_rate = anneal_rate
        self.steps = 0

        # replay memory
        self.rlmemory = ReplayMemory(replay_memory_size)
        self.slmemory = SLMemory(sl_memory_size)
        self.replay_start_size = replay_start_size
        

    
    '''def greedy_policy(self,state,step):
        """Epsilon-greedy policy for training, greedy policy otherwise."""
        explore_prob = self.max_explore - (step * self.anneal_rate)
        explore = max(explore_prob, self.min_explore) > np.random.rand() 
        if(step > 25000):
            explore = False  

        if explore:
            action = np.random.randint(0,self.action_space_size-1)
        else:              
            state   = Variable(torch.FloatTensor(state).unsqueeze(0).to(device), volatile=True)              
            qvalues = self.online_network.forward(state) #,1, 1, hidden_s, cell_s)
            action  = qvalues.max(1)[1].data[0]
        return action   

    def avg_policy(self,state,step): 
        state   = Variable(torch.FloatTensor(state).unsqueeze(0).to(device), volatile=True)
        distribution= self.policy_network.forward(state)    
        dist = Categorical(distribution)        
        action = dist.sample().item()        
           
        return action'''
    
    
    def policy(self,state,step,is_best,episode): 
        """Epsilon-greedy policy for training, greedy policy otherwise."""
        explore_prob = self.max_explore - (step * self.anneal_rate)
        explore = max(explore_prob, self.min_explore) > np.random.rand() 
        if(step > 25000):
            explore = False     
           
        if(random.random() > self.eta):
              state   = Variable(torch.FloatTensor(state).unsqueeze(0).to(device), volatile=True)
              distribution= self.policy_network.forward(state)

              sm = distribution[0].tolist()
              for item in sm:
                  if(item != item or item == float('inf')):
                      print("SOFTMAX ANOMALY AT EPISODE",episode)
              if(sum(sm) == 0):
                      print("SOFTMAX ANOMALY AT EPISODE",episode)   
                    
              
              #action = torch.multinomial(distribution,1).item()
            
              dist = Categorical(distribution)
              
              action = dist.sample().item()
             

              #action = np.where(dist == np.amax(dist))
              
              

        else:     

          if explore:

              action = np.random.randint(0,self.action_space_size-1)
          else:
              is_best = True

              state   = Variable(torch.FloatTensor(state).unsqueeze(0).to(device), volatile=True)
              
              qvalues = self.online_network.forward(state) #,1, 1, hidden_s, cell_s)
              action  = qvalues.max(1)[1].data[0]

         
        return action, is_best

    def plot_grad_flow(self,named_parameters,name,episode):
        ave_grads = []
        layers = []
        for n, p in named_parameters:
            if(p.requires_grad) and ("bias" not in n):
                layers.append(n)
                ave_grads.append(p.grad.abs().mean())

               
 
        plt.plot(ave_grads, alpha=0.3, color="b")
        plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
        plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
        plt.xlim(xmin=0, xmax=len(ave_grads))
        plt.xlabel("Layers")
        plt.ylabel("average gradient")
        plt.title("Gradient flow " + name)
        plt.savefig('nfsp_data/gradients ' + name + str(episode) + '.png')
      
       
        plt.grid(True)
        plt.close()


        
    def plot_gradients(self,name,episode):
      
        self.plot_grad_flow(self.policy_network.named_parameters(),name,episode)
        
        
    def update_target_network(self):
        """Update target network weights with current online network values."""
        self.target_network.load_state_dict(self.online_network.state_dict()) 


    def train_rl_network(self,step):
        """Update online network weights."""
        batch = self.rlmemory.sample(self.batch_size)
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
        rl_loss = (q_value - Variable(expected_q_value.data)).pow(2).mean()
            
        self.rloptimizer.zero_grad()
        rl_loss.backward()
        self.rloptimizer.step()

        if step % self.target_update_freq == 0:
            self.update_target_network() 
    
    
    def train_sl_network(self):
        """Update online network weights."""
        states, actions = self.slmemory.sample(self.batch_size)




        
        state      = Variable(torch.FloatTensor(np.float32(states))).to(device)
        action     = Variable(torch.LongTensor(actions)).to(device)


     
        probs = self.policy_network(state)
        probs_with_actions = probs.gather(1, action.unsqueeze(1))
        log_probs = probs_with_actions.log()

        sl_loss = -1 * log_probs.mean()
    

            
        self.sloptimizer.zero_grad()
        sl_loss.backward()
        self.sloptimizer.step()


                        
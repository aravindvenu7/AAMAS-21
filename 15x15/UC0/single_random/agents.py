from networks_maddpg import Network, Critic_Network
from storage_maddpg import ReplayBuffer
import torch, random
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical
import numpy as np
from misc import hard_update, gumbel_softmax, onehot_from_logits
from noise import OUNoise

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




class DDPGAgent(object):
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
        self.critic_state_size = 5*self.grid_size*self.grid_size*self.state_space_size + 3*15 + 2*5
        self.policy = Network(self.state_space_size, self.action_space_size)
        self.critic = Critic_Network(self.state_space_size*5, 3*15 + 2*5, 1)
        self.target_policy = Network(self.state_space_size, self.action_space_size)
        self.target_critic = Critic_Network(self.state_space_size*5, 3*15 + 2*5, 1)
     
        self.policy = self.policy.to(device)
        self.critic = self.critic.to(device)
        self.target_policy = self.target_policy.to(device)
        self.target_critic = self.target_critic.to(device)
        
        #self.update_target_network()
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=0.0003)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.0003)
        # training parameters
        #self.target_update_freq = target_update_freq
        self.gamma = discount
        self.tau = 0.01
        #self.discount = discount
        #self.batch_size = batch_size
        hard_update(self.target_policy, self.policy)
        hard_update(self.target_critic, self.critic)
        print("agent set up")
        self.exploration = 0.3

        # policy during learning
        #self.max_explore = max_explore #+ (anneal_rate * replay_start_size)
        #self.min_explore = min_explore
        #self.anneal_rate = anneal_rate
        #self.steps = 0

        # replay memory
        #self.memory = Memory(replay_memory_size)
        #self.replay_start_size = replay_start_size
        #self.experience_replay = Memory(replay_memory_size)

    def reset_noise(self):
        if not self.discrete_action:
            self.exploration.reset()

    def scale_noise(self, scale):
        if self.discrete_action:
            self.exploration = scale
        else:
            self.exploration.scale = scale    

    def get_params(self):
        return {'policy': self.policy.state_dict(),
                'critic': self.critic.state_dict(),
                'target_policy': self.target_policy.state_dict(),
                'target_critic': self.target_critic.state_dict(),
                'policy_optimizer': self.policy_optimizer.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict()}

    def load_params(self, params):
        self.policy.load_state_dict(params['policy'])
        self.critic.load_state_dict(params['critic'])
        self.target_policy.load_state_dict(params['target_policy'])
        self.target_critic.load_state_dict(params['target_critic'])
        self.policy_optimizer.load_state_dict(params['policy_optimizer'])
        self.critic_optimizer.load_state_dict(params['critic_optimizer'])

    def step(self, obs, explore=False):
       
        """
        Take a step forward in environment for a minibatch of observations
        Inputs:
            obs (PyTorch Variable): Observations for this agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            action (PyTorch Variable): Actions for this agent
        """

        action = self.policy(obs)
        
        if explore:
                action = gumbel_softmax(action, hard=True)
        else:
                action = onehot_from_logits(action)

        return action




            
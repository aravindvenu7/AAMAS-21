import numpy as np
import torch
import matplotlib.pyplot as plt
%matplotlib inline
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch import float32
from time import time
from torch.distributions import Normal
from train_dqn import main, evaluate, load_models
import gym
import make_env
from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios
from copg_optim import CoPG
import random
gridsize = 15
def_pos = []
defender_rewards = []
poacher_rewards = []
opt_allocations = []
def_pos = []
poacher_pos = []
def_visits = []
poacher_visits = []

class Config(object):


  def __init__(self):
    self.save_model = False
    self.dtype = float32
    self.dtype_long = torch.long
    self.debug = False
    self.true_embeddings = True
    self.embed_lr = 1e-4
    self.reduced_action_dim = 10
    self.buffer_size = 20
    self.initial_phase_epochs = 1000
    self.device = 'cuda'
    self.emb_flag = 'exec'
    self.TIS = False #Importance sampling
    self.emb_lambda = 1.0
    self.save_after = 50000
    self.restore = False
    self.max_episodes = 10000
    self.initial_phase_epochs = 500
    self.feature_dim = [256]
    self.actor_lr = 1e-3
    self.critic_lr = 1e-2
    self.state_lr = 1e-3
    self.gauss_variance = -1
    self.fourier_order = 0


class Config_poacher(object):


  def __init__(self):
    self.save_model = False
    self.dtype = float32
    self.dtype_long = torch.long
    self.debug = False
    self.true_embeddings = True   #True for pre-trained embeddings
    self.embed_lr = 1e-4
    self.reduced_action_dim = 2  #Embedding size
    self.buffer_size = 20    #Supervised memory size for learning embeddings
    self.initial_phase_epochs = 1000
    self.device = 'cuda'
    self.emb_flag = 'exec'   #Whether to choose sampled embedding output by algorithm or to use the embedding out of 20k which is at least distance from it
    self.TIS = False #Importance sampling
    self.emb_lambda = 1.0
    self.save_after = 50000
    self.restore = False
    self.max_episodes = 10000
    self.initial_phase_epochs = 500  #For learning embeddings only
    self.feature_dim = [32]  #Dimensionality of features output by the state feature network. These go as input to the actor and critic
    self.actor_lr = 1e-3  #Acor learning rate
    self.critic_lr = 1e-2
    self.state_lr = 1e-3  #For network that takes state as input and outputs state features
    self.gauss_variance = -1
    self.fourier_order = 0
    self.state_dim = 225*2
    self.action_dim = 225

class Action_representation():
    def __init__(self,config,embeddins):

        #self.state_dim = 225*2 #The dimensionality of state features
        #self.action_dim = 10000
        self.config = config


            
        embeddings = embeddins           
        embeddings = np.expand_dims(embeddings, axis=0)
        maxi, mini = np.max(embeddings), np.min(embeddings)
        embeddings = ((embeddings - mini)/(maxi-mini))*2 - 1  # Normalize to (-1, 1)
        self.embeddings = Variable(torch.from_numpy(embeddings).type(self.config.dtype), requires_grad=False)
        self.reduced_action_dim = np.shape(embeddings)[2]  #Latent dimensionality; Embedding shape : (1, n_actions,latent_dims)



    def get_match_scores(self, action):

        action = action.unsqueeze(1)  #predicted embedding
        embeddings = self.embeddings
        diff = torch.norm(embeddings - action, p=2, dim=-1)
        return diff


    def get_best_match(self, action):    #action here is the predicted embedding
        diff = self.get_match_scores(action)       
        val, pos = torch.min(diff, dim=1)
        
       
        return pos.cpu().data.numpy()[0]


    def get_match_dist(self, action):
        diff = self.get_match_scores(action)
        probs = F.softmax(-diff, dim=-1)       # probs = F.softmax(1.0/(diff+1e-10), dim=-1)

  
        return probs


    def get_embedding(self, action):
        # Get the corresponding target embedding
        action_emb = self.embeddings[:, action]    
        return action_emb

class Actor(nn.Module):

    def __init__(self, feature_dim, action_dim, varf):
        super(Actor, self).__init__()


        #self.conv1 = nn.Conv2d(num_inputs,10, 3,padding=1)
        self.linear1 = nn.Linear(225, feature_dim)
        self.mean = nn.Linear(feature_dim, action_dim)
        self.variance = nn.Linear(feature_dim, action_dim)
        self.varf = varf
 

    def forward(self, x):
        x = Variable(torch.from_numpy(x).float(), requires_grad=False)

        #x = F.relu(self.conv1(x))
        #x = x.view(-1,4500)
        x = F.tanh(self.linear1(x))
        mean = F.tanh(self.mean(x))
        var  = F.sigmoid(self.variance(x)) + self.varf
        #var = torch.ones_like(mean, requires_grad=False) * self.varf
        return mean, var
        
        
    def get_action_1(self, state):
        mean, var = self.forward(state)
        dist = Normal(mean, var)
        action = dist.sample()
        # action = torch.clamp(action, -1, 1) #DONT DO THIS

        return action

    def get_log_prob(self, state, action1):
     
        mean, var = self.forward(state)
        dist = Normal(mean, var)
        #print(mean.size(), var.size(), action1.size())

        return dist.log_prob(action1)       

class Critic(nn.Module):
    def __init__(self, feature_dim):
        super(Critic, self).__init__()

        self.critic = nn.Sequential(nn.Linear(225, feature_dim), 
                                    nn.Tanh(),
                                    nn.Linear(feature_dim, 1))  

     
    def forward(self, state):
        state = Variable(torch.from_numpy(state).float(), requires_grad=False)
        value = self.critic(state)
        return value

class CGDAgent(object):
    """Deep Q-learning agent."""

    def __init__(self,config_d, config_p, embeddings_d, embeddings_p, feature_dim_d, action_dim_d, feature_dim_p, action_dim_p):
        """Set parameters, initialize network."""
    
        self.actor_d = Actor(feature_dim_d, action_dim_d, 1e-2)  #4e-1
        self.actor_p = Actor(feature_dim_p, action_dim_p, 1e-2)

        self.critic = Critic(feature_dim_d)

        self.action_rep_d = Action_representation(config_d,embeddings_d)        
        self.action_rep_p = Action_representation(config_p,embeddings_p)

        self.optim_actor =  CoPG(self.actor_d.parameters(),self.actor_p.parameters(), lr =4e-5)
        self.optim_critic = torch.optim.Adam(self.critic.parameters(), lr=1e-2)

         
        self.ep_rewards1 = []
        self.ep_rewards2 = []
        self.ep_states = []
        self.ep_actions1 = []
        self.ep_actions2 = []
        self.ep_exec_action_embs1 = []
        self.ep_chosen_action_embs1 = []
        self.ep_exec_action_embs2 = []
        self.ep_chosen_action_embs2 = []

        self.atype = config_d.dtype
        self.config_d = config_d
        self.config_p = config_p


    

    def get_action_d(self, state):            
                state = np.float32(state)
                if len(state.shape) == 1:
                    state = np.expand_dims(state, 0)

                
                chosen_action_emb = self.actor_d.get_action_1(state)                
                action = self.action_rep_d.get_best_match(chosen_action_emb)          
                exec_action_emb = self.action_rep_d.get_embedding(action).cpu().view(-1).data.numpy()             
                chosen_action_emb = chosen_action_emb.cpu().view(-1).data.numpy()
                return action, (exec_action_emb, chosen_action_emb)
    

    def get_action_p(self, state):            
                state = np.float32(state)
                if len(state.shape) == 1:
                    state = np.expand_dims(state, 0)

                
                chosen_action_emb = self.actor_p.get_action_1(state)                
                action = self.action_rep_p.get_best_match(chosen_action_emb)          
                exec_action_emb = self.action_rep_p.get_embedding(action).cpu().view(-1).data.numpy()             
                chosen_action_emb = chosen_action_emb.cpu().view(-1).data.numpy()
                return action, (exec_action_emb, chosen_action_emb)        
        
    def update(self, s, a1, a2, a_emb1, a_emb2, r1, r2):  
        
                # Store the episode history
                for i in range(len(s)):
                  self.ep_rewards1.append(r1[i])
                  self.ep_rewards2.append(r2[i])
                  self.ep_states.append(s[i])
                  self.ep_actions1.append(int(a1[i]))
                  self.ep_actions2.append(int(a2[i]))
                  self.ep_exec_action_embs1.append(a_emb1[i][0])
                  self.ep_chosen_action_embs1.append(a_emb1[i][1])
                  self.ep_exec_action_embs2.append(a_emb2[i][0])
                  self.ep_chosen_action_embs2.append(a_emb2[i][1])
                           



                self.optimize(np.float32(self.ep_states), np.float32(self.ep_actions1), np.float32(self.ep_actions2), np.float32(self.ep_exec_action_embs1), np.float32(self.ep_chosen_action_embs1),
                              np.float32(self.ep_exec_action_embs2), np.float32(self.ep_chosen_action_embs2), np.float32(self.ep_rewards1), np.float32(self.ep_rewards2))

                # Reset the episode history
                self.ep_rewards1 = []
                self.ep_rewards2 = []
                self.ep_states = []
                self.ep_actions1 = []
                self.ep_actions2 = []
                self.ep_exec_action_embs1 = []
                self.ep_chosen_action_embs1 = []
                self.ep_exec_action_embs2 = []
                self.ep_chosen_action_embs2 = []
              




    '''def check_nan(self):
        # Check for nan periodically
        self.ctr += 1
        if self.ctr == self.nan_check_fequency:
            self.ctr = 0
            # Note: nan != nan  #https://github.com/pytorch/pytorch/issues/4767
            for name, param in self.named_parameters():
                if (param != param).any():
                    raise ValueError(name + ": Weights have become nan... Exiting.")'''


    def optimize(self, s, a1, a2, exec_a1_emb, chosen_a1_emb,  exec_a2_emb, chosen_a2_emb, r1, r2): 
            r1 = Variable(torch.from_numpy(r1).type(self.config_d.dtype), requires_grad=False).view(-1, 1)          
            r2 = Variable(torch.from_numpy(r2).type(self.config_d.dtype), requires_grad=False).view(-1, 1)
            exec_a1_emb = Variable(torch.from_numpy(exec_a1_emb).type(self.config_d.dtype), requires_grad=False)
            chosen_a1_emb = Variable(torch.from_numpy(chosen_a1_emb).type(self.config_d.dtype), requires_grad=False)
            exec_a2_emb = Variable(torch.from_numpy(exec_a2_emb).type(self.config_d.dtype), requires_grad=False)
            chosen_a2_emb = Variable(torch.from_numpy(chosen_a2_emb).type(self.config_d.dtype), requires_grad=False)            
            a1_emb = exec_a1_emb if self.config_d.emb_flag == 'exec' else chosen_a1_emb
            a2_emb = exec_a2_emb if self.config_d.emb_flag == 'exec' else chosen_a2_emb

            # ---------------------- optimize critic ----------------------
            val_pred = self.critic.forward(s)
            # loss_baseline = F.smooth_l1_loss(val_pred, r1)
            loss_critic = F.mse_loss(val_pred, r1)
            
            td_error = (r1 - val_pred).detach().transpose(0,1)  #Might need to rescale reward

            #print("td errors calculated")
          
            log_probs1_inid = self.actor_d.get_log_prob(s, a1_emb)
            log_probs1 = log_probs1_inid.sum(1)
         
            log_probs2_inid = self.actor_p.get_log_prob(s, a2_emb)
            log_probs2 = log_probs2_inid.sum(1)
            #print(log_probs1.mean(), log_probs2.mean())
            #print("log_probs calculated")
            
            #----------------------critic update---------------------------
            self.optim_critic.zero_grad()
            loss_critic.backward()
            #torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10) 
            self.optim_critic.step()
            #print("critic update done")
            
            
            #----------------------actor update----------------------------
            r1 = r1.transpose(0,1)
            #td_error = td_error.to(device)
            objective = log_probs1*log_probs2*(td_error)
            #print("Objective is")
            ob = objective.mean()
            #print(ob)
            lp1 = log_probs1*(td_error)
            lp1=lp1.mean()
            lp2 = log_probs2*(td_error)
            lp2=lp2.mean()
            #print("lp1 and lp2 are", lp1, lp2)

            #loss_actor = -1.0 * torch.mean(td_error * self.actor.get_log_prob(s1, a1_emb))


            self.optim_actor.zero_grad()
            #torch.nn.utils.clip_grad_norm_(self.actor_d.parameters(), 10)
            #torch.nn.utils.clip_grad_norm_(self.actor_p.parameters(), 10)
            self.optim_actor.step(ob, lp1,lp2)
            #print("actor update done")
 
            




def main():
    def_pos = []
    defender_rewards = []
    poacher_rewards = []
    opt_allocations = []
    def_pos = []
    poacher_pos = []
    def_visits = []
    poacher_visits = []
    poacher_reward_mean = 0
    env = make_env.make_env("simple_tag")
    
 
    config = Config()
    configp = Config_poacher()
    agent_d, agent_r = load_models()  #LOAD ALL NETWORKS
    print("DQNs loaded")
    
    allocations = np.load('actions.npy').reshape((-1,15,15))
    d_allocations = []
      
    for i in range(10000):
        



        alloc = []
        inits = []

        for x in range(15):
          for y in range(15):

            if(allocations[i][x][y] == 1):
              inits.append(np.array([x,y]))


        d_allocations.append(inits)    

    

    #INDEXING POACHER ACTIONS
    p_allocations = []
    for i in range(gridsize):
      for j in range(gridsize):
        p_allocations.append(np.array([i,j]))
   
    print("Allocations loaded", len(d_allocations), len(p_allocations))

    defender_embeddings = np.load('action_embeddings.npy')
    defender_embeddings = defender_embeddings[:10000]

    poacher_embeddings = []
    for i in range(gridsize):
      for j in range(gridsize):
        poacher_embeddings.append([i/14, j/14])
    poacher_embeddings = np.array(poacher_embeddings) 

    




    
  

    cgdagent = CGDAgent(config, configp, defender_embeddings, poacher_embeddings, 128, 50, 32, 2)
    
    
    #state = np.zeros((15,15,2))

    defender_counts = np.zeros((15,15))

    #state[:,:,0] = env.world.animal_densities.copy()
    #state[:,:,1] = defender_counts.copy()
    state = env.world.animal_densities.copy()
    state = state.reshape((225))
    poacher_index = []
    batch_size = 10
    
    

    for trial in range(5000):
      #print("TRIAL NO:", trial)
     
      
      d_rewards = []
      p_rewards = [] 
      defender_states = []
      poacher_states = []
      defender_actions = []
      poacher_actions = []
      defender_action_embed = []
      poacher_action_embed = []

      #DATA COLLECTION
      #alpha = np.random.random()
      alpha = 1.0
      for episode in range(batch_size):
        
        
        defender_inits = []
        poacher_inits = []          
        
    
        
        #SAMPLE DEFENDER ALLOCATION
        defender_index, extra_info = cgdagent.get_action_d(state)
        
        if(alpha<0.05):
          defender_index = np.random.randint(0,len(d_allocations))
          extra_info = (defender_embeddings[defender_index], defender_embeddings[defender_index])


        defender_inits = d_allocations[defender_index]
        defender_actions.append(defender_index)   
        defender_action_embed.append(extra_info)    #change
        def_pos.append(defender_inits)




        #SAMPLE POACHER ALLOCATION
        poacher_state = state.copy()

        defender_states.append(state)
        poacher_states.append(poacher_state)


        poacher_index,p_extra_info = cgdagent.get_action_p(poacher_state)
        
        if(alpha<0.05):
          poacher_index = np.random.randint(0,len(p_allocations))
          p_extra_info = (poacher_embeddings[poacher_index], poacher_embeddings[poacher_index])
        
        poacher_inits = p_allocations[poacher_index]
    
       
        poacher_inits = [np.array([poacher_inits[0], poacher_inits[1]])]    
        poacher_pos.append(poacher_inits)
        poacher_actions.append(poacher_index)
        poacher_action_embed.append(p_extra_info)    
      
        #if(trial>150):
          #print(defender_inits)
          #print(poacher_inits[0])
        #EVALUATE ACTION

        reward, d_visits, p_visits = evaluate(defender_inits, poacher_inits, agent_d, agent_r, random = False)  #evaluate function here
        reward = reward/1000
        def_visits.append(d_visits)
        poacher_visits.append(p_visits)
        
        poacher_reward = -1 * reward
        defender_rewards.append(reward)
        poacher_rewards.append(poacher_reward)

        d_rewards.append(100+reward)
        p_rewards.append(-1*(100+reward))

        #UPDATE DEFENDER COUNTS
        for item in defender_inits:
          defender_counts[int(item[0])][int(item[1])] += 1


        #UPDATE STATE
        #new_state = np.zeros((15,15,2))
        new_state = env.world.animal_densities.copy()
        #new_state[:,:,1] = defender_counts.copy()
        new_state = new_state.reshape((225))
        poacher_new_state = new_state.copy()

        state = new_state





        #MODEL UPDATION
      #print("data collection done")
      cgdagent.update(defender_states , defender_actions, poacher_actions, defender_action_embed, poacher_action_embed, d_rewards, p_rewards )
      #print("update done")


      

      '''if(trial%10==0 and trial > 0):
          plt.plot(defender_rewards)
          plt.show()
          plt.plot(poacher_rewards)
          plt.show()'''
      if(len(poacher_rewards) == 5000):
          np.save('coPG_results/defender_rewards5.npy', np.array(defender_rewards))
          np.save('coPG_results/poacher_rewards5.npy', np.array(poacher_rewards))
          np.save('coPG_results/defender_allocations5.npy', np.array(def_pos))
          np.save('coPG_results/poacher_allocations5.npy', np.array(poacher_pos))
          np.save('coPG_results/defender_visits5.npy', np.array(def_visits))
          np.save('coPG_results/poacher_visits5.npy', np.array(poacher_visits))    

            
if __name__ == '__main__':
    main()



import torch
import torch.nn.functional as F
from misc import soft_update, average_gradients, onehot_from_logits, gumbel_softmax
from agents import DDPGAgent
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MSELoss = torch.nn.MSELoss()

class MADDPG(object):
    """
    Wrapper class for DDPG-esque (i.e. also MADDPG) agents in multi-agent task
    """
    def __init__(self, gamma=0.95, tau=0.01, lr=0.01):
        """
        Inputs:
            agent_init_params (list of dict): List of dicts with parameters to
                                              initialize each agent
                num_in_pol (int): Input dimensions to policy
                num_out_pol (int): Output dimensions to policy
                num_in_critic (int): Input dimensions to critic
            alg_types (list of str): Learning algorithm for each agent (DDPG
                                       or MADDPG)
            gamma (float): Discount factor
            tau (float): Target update rate
            lr (float): Learning rate for policy and critic
            hidden_dim (int): Number of hidden dimensions for networks
            discrete_action (bool): Whether or not to use discrete action space
        """
        agent_d = DDPGAgent(
                      name = "drone",
                      action_space_size = 15,
                      channels = 8,
                      target_update_freq= 20,  #20
                      discount=0.99,
                      batch_size=32,
                      max_explore=1,
                      min_explore=0.05,
                      anneal_rate=(9 / 10000),
                      replay_memory_size=320*6,
                      replay_start_size=32*6)


        agent_r = DDPGAgent(
                      name = "ranger",
                      action_space_size = 5,
                      channels = 8,
                      target_update_freq= 50,  #20
                      discount=0.99,
                      batch_size=32,
                      max_explore=1,
                      min_explore=0.05,
                      anneal_rate=(9 / 10000),
                      replay_memory_size=320*2,
                      replay_start_size=32*2) 

        self.agents = [agent_d, agent_r]   #todo
        print("Number of agents",len(self.agents))               
        self.discrete_action= True
        self.gamma = gamma
        self.tau = tau                                                                                                                                                                                               
        self.lr = lr




    def scale_noise(self, scale):
        """
        Scale noise for each agent
        Inputs:
            scale (float): scale of noise
        """
        for a in self.agents:
            a.scale_noise(scale)

    def reset_noise(self):
        for a in self.agents:
            a.reset_noise()

    '''def step(self, observations, explore=False):
        """
        Take a step forward in environment with all agents
        Inputs:
            observations: List of observations for each agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            actions: List of actions for each agent
        """
        return [a.step(obs, explore=explore) for a, obs in zip(self.agents,
                                                                 observations)]'''

    def update(self, sample, agent_i, parallel=False, logger=None):
        """
        Update parameters of agent model based on sample from replay buffer
        Inputs:
            sample: tuple of (observations, actions, rewards, next
                    observations, and episode end masks) sampled randomly from
                    the replay buffer. Each is a list with entries
                    corresponding to each agent
            agent_i (int): index of agent to update
            parallel (bool): If true, will average gradients across threads
            logger (SummaryWriter from Tensorboard-Pytorch):
                If passed in, important quantities will be logged
        """
        obs, acs, rews, next_obs, dones = sample
        curr_agent = self.agents[agent_i]
        

        curr_agent.critic_optimizer.zero_grad()
        all_trgt_acs = []
        for i in range(5):
            if(i<=2):
                ac = onehot_from_logits(self.agents[0].target_policy(next_obs[i].view(-1,8,15,15)))
            else:
                ac = onehot_from_logits(self.agents[1].target_policy(next_obs[i].view(-1,8,15,15)))   
            all_trgt_acs.append(ac.cpu())     

        
        next_obs1 = np.hstack(next_obs)    
        next_obs1 = next_obs1.reshape((-1,40,15,15))  
 
        all_trgt_acs = np.hstack(all_trgt_acs)
        
           
        #trgt_vf_in = torch.cat((*next_obs, *all_trgt_acs), dim=1)
        if(curr_agent.name == "drone"):
            rand = np.random.randint(3)
        else:
            rand =  np.random.randint(3,5)
           
         
        target_value = (rews[rand].view(-1, 1).to(device) + self.gamma *
                        curr_agent.target_critic(next_obs1, all_trgt_acs) *
                        (1 - dones[rand].view(-1, 1).to(device)))
       
        
        #vf_in = torch.cat((*obs, *acs), dim=1)
        obs1 = np.hstack(obs)    
        obs1 = obs1.reshape((-1,40,15,15))    
        acs = np.hstack(acs)

        actual_value = curr_agent.critic(obs1, acs)

        vf_loss = MSELoss(actual_value, target_value.detach())
        vf_loss.backward()
        if parallel:
            average_gradients(curr_agent.critic)
        torch.nn.utils.clip_grad_norm(curr_agent.critic.parameters(), 0.5)
        curr_agent.critic_optimizer.step()
   
        curr_agent.policy_optimizer.zero_grad()


        # Forward pass as if onehot (hard=True) but backprop through a differentiable
        # Gumbel-Softmax sample. The MADDPG paper uses the Gumbel-Softmax trick to backprop
        # through discrete categorical samples, but I'm not sure if that is
        # correct since it removes the assumption of a deterministic policy for
        # DDPG. Regardless, discrete policies don't seem to learn properly without it.
        curr_pol_out = curr_agent.policy(obs[rand])
        curr_pol_vf_in = gumbel_softmax(curr_pol_out, hard=True)

        
        
        all_pol_acs = []
       
        for i in range(5):
                if i == rand:
                    all_pol_acs.append(curr_pol_vf_in)
                else:
                    if(i<=2):
                        ac = onehot_from_logits(self.agents[0].target_policy(next_obs[i]))
                    else:
                        ac = onehot_from_logits(self.agents[1].target_policy(next_obs[i]))  
                    all_pol_acs.append(ac)
                    

        #vf_in = torch.cat((*obs, *all_pol_acs), dim=1)
        obs2 = np.hstack(obs)    
        obs2 = obs2.reshape((-1,40,15,15)) 
        all_pol_acs = torch.hstack(all_pol_acs)
       
        pol_loss = -curr_agent.critic(obs2, all_pol_acs).mean()
        pol_loss += (curr_pol_out**2).mean() * 1e-3
        pol_loss.backward()
     
        if parallel:
            average_gradients(curr_agent.policy)
        torch.nn.utils.clip_grad_norm(curr_agent.policy.parameters(), 0.5)
        curr_agent.policy_optimizer.step()



    def update_all_targets(self):
        """
        Update all target networks (called after normal updates have been
        performed for each agent)
        """
        for a in self.agents:
            soft_update(a.target_critic, a.critic, self.tau)
            soft_update(a.target_policy, a.policy, self.tau)
        


import gym   #might run into version issue
import pickle
import make_env
from agent_a2c2 import RLAgent
from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios
import sys
import numpy as np  
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import pandas as pd
from IPython import display as ipythondisplay
from collections import deque
import random
import cv2
from scipy.stats import rankdata
import itertools, math
from torch.distributions import Categorical, Multinomial
seed = 0
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
print("Seed 0")   


def process_states(dronepos,rangerpos,dronesig,dronewarn,densities,obs,env):
  states = []
  agentpos = dronepos + rangerpos
  den = np.zeros((15,15))
  obv = np.zeros((15,15))

  i = 0
  
  for pos in agentpos:
    #print(pos)
    den[int(pos[0])][int(pos[1])] = densities[i]
    i = i+1
  i = 0
  for pos in dronepos:
    #print(pos)
    obv[int(pos[0])][int(pos[1])] = obs[i]
    i = i+1

  for i in range(len(agentpos)):
    state = np.zeros((15,15,8))
    a = np.zeros((15,15))
    b = np.zeros((15,15))
    c = np.zeros((15,15))
    d = np.zeros((15,15))
    e = np.zeros((15,15))
    
    a[int(agentpos[i][0])][int(agentpos[i][1])] = 1.0
    for i in range(len(dronepos)):
      b[int(dronepos[i][0])][int(dronepos[i][1])] = 1.0
      d[int(dronepos[i][0])][int(dronepos[i][1])] = dronesig[i]
      e[int(dronepos[i][0])][int(dronepos[i][1])] = dronewarn[i]
    for pos in rangerpos:
      c[int(pos[0])][int(pos[1])] = 1.0
    state[:,:,0] = a
    #print("A",a[:,:])
    state[:,:,1] = b 
    #print("B",b[:,:])
    state[:,:,2] = c 
    #print("C",c[:,:])
    state[:,:,3] = obv
    state[:,:,4] = d
    #print("D",d[:,:])
    state[:,:,5] = e
    #print("E",e[:,:])
    state[:,:,6] = den
    if(np.amax(env.world.v_counts == 0)):
      state[:,:,7] = np.zeros((15,15))
    else:  
      state[:,:,7] = env.world.v_counts/ np.sum(env.world.v_counts)
    state = state.transpose(2,0,1)
    states.append(state)
  return states






def plot_rewards(rewards_d,rewards_r,episode, counts, winrates):
    rewards1 = []

    for i in range (len(rewards_d)):
      rewards1.append(rewards_r[i] + rewards_d[i])
    avg_rew = []
    j = 0
    while(j < len(rewards_d) - 20):

      x = rewards1[j:j+20]
      sum1 = np.sum(np.array(x)) /20.0
      avg_rew.append(sum1)
      j = j+1

    counts1 = counts[-1].reshape((225))
    countrank1 = rankdata(counts1,method = 'ordinal') 
    countrank1 = (225 - countrank1) / 225.0  
    coverage = countrank1.reshape((15,15)) 
    
    if(episode > 0):
      counts2 = counts[-100].reshape((225))
      countrank2 = rankdata(counts2,method = 'ordinal') 
      countrank2 = (225 - countrank2) / 225.0  
      prev_coverage = countrank2.reshape((15,15))     
    else:
      prev_coverage = np.zeros((15,15))


    plt.plot(avg_rew)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.savefig('a2c2_data/rewards_dqn_ep' + str(episode) + '.png')
    plt.close()

    plt.plot(winrates)
    plt.xlabel('Episode*100')
    plt.ylabel('Winrate')
    plt.savefig('a2c2_data/winrates_dqn_ep' + str(episode) + '.png')
    plt.close()

    plt.xlabel('a2c2_data/Visitation counts')
    plt.imshow(counts[-1])
    plt.savefig('a2c2_data/v_counts_dqn_ep' + str(episode) + '.png')
    plt.close()

    plt.xlabel('Attacker probability acc to coverage')
    plt.imshow(coverage)
    plt.savefig('a2c2_data/attacker_prob_dqn_ep' + str(episode) + '.png')
    plt.close()


 
    plt.xlabel('Coverage - Prev. Coverage')
    x = abs(coverage - prev_coverage)
    plt.imshow(x)
    plt.savefig('a2c2_data/coverage_diff_dqn_ep' + str(episode) + '.png')
    plt.close()


    if(episode > 0 ):
      y = abs(counts[-1] - counts[-100])
      plt.xlabel('Count - Prev. Count')
      plt.imshow(y)
      plt.savefig('a2c2_data/counts_diff_dqn_ep' + str(episode) + '.png')
      plt.close()



def load_models():
      agent_d = RLAgent(
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


      agent_r = RLAgent(
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
     
      
      
      agent_d.online_network.load_state_dict(torch.load("models/drone_online60000.pth"))
      agent_d.target_network.load_state_dict(torch.load("models/drone_target60000.pth"))      
      agent_r.online_network.load_state_dict(torch.load("models/ranger_online60000.pth"))      
      agent_r.target_network.load_state_dict(torch.load("models/ranger_target60000.pth"))
        

      return agent_d, agent_r    


def evaluate(defender_inits, poacher_inits, agent_d, agent_r, random = False):
      
      env = make_env.make_env("simple_tag")

      visitation = []
      winrates = []
      '''agent_d = RLAgent(
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


      agent_r = RLAgent(
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
     
      
      
      agent_d.online_network.load_state_dict(torch.load("uc_models/drone_online60000.pth"))
      agent_d.target_network.load_state_dict(torch.load("uc_models/drone_target60000.pth"))      
      agent_r.online_network.load_state_dict(torch.load("uc_models/ranger_online60000.pth"))      
      agent_r.target_network.load_state_dict(torch.load("uc_models/ranger_target60000.pth"))'''    
      def_allocation_counts = np.full((env.world.gridsize, env.world.gridsize), 1)
      rewards = []
      rewards_r = []
      rewards_d = []
      
      coverages = []
      steps = 500000
      winrate = 0
      ep_actions = []
      for episode in range(1):
          
      
          
          # Code below is for random inits
          if(random):  
            inits = []
            
            init0 = np.random.choice(env.world.gridsize, 5, replace = False )
            init1 = np.random.choice(env.world.gridsize, 5, replace = False )
            for i in range(5):
              inits.append(np.array([init0[i], init1[i]]))
            drone_inits = inits[:-2]
            ranger_inits = inits[-2:]  


          
            poacher_inits = [np.random.randint(0,env.world.gridsize, size = env.world.dim_p)]
          else:  
            drone_inits = defender_inits[:-2]
            ranger_inits = defender_inits[-2:]
          
          coverages.append(env.world.v_counts.copy())
          for item in  drone_inits:
            def_allocation_counts[int(item[0])][int(item[1])] += 1
          for item in  ranger_inits:
            def_allocation_counts[int(item[0])][int(item[1])] += 1  

          #print("Defender init:",defender_inits)
          #print("Poacher init:",poacher_inits)      
          
          obs = env.reset(drone_inits, ranger_inits, poacher_inits)  #returns 0 or 1
          
          
          drone_pos_old = []
          drone_signals_old = []
          drone_warnings_old = []
          ranger_pos_old = []
          animal_densities_old = []
          

        
          ######### PROCESSING STATE  ######################
          for drone in env.world.drones:
          
            drone_pos_old.append(drone.state.p_pos)
            drone_signals_old.append(drone.state.signal)
            drone_warnings_old.append(drone.state.warn)
            animal_densities_old.append(env.world.pfr_temp[int(drone.state.p_pos[0])][int(drone.state.p_pos[1])])
          for ranger in env.world.rangers:
            ranger_pos_old.append(ranger.state.p_pos)
            poacher_pos_old = env.world.poachers[0].state.p_pos
            animal_densities_old.append(env.world.pfr_temp[int(ranger.state.p_pos[0])][int(ranger.state.p_pos[1])]) 

          
          states = process_states(drone_pos_old,ranger_pos_old,drone_signals_old,drone_warnings_old,animal_densities_old,obs[:-2],env)
          dstates = states[:-2]
          rstates = states[-2:]

          ######################################################



          
          episode_reward_drone = 0
          episode_reward_ranger = 0


          
          for step in range(100):
     
              steps = steps + 1
              d_actions = []
              r_actions = []
              actions = []
              d_rewards = []
              r_rewards = []
              for i in range(len(dstates)):
                #########################  new  #################################
                action = agent_d.policy(dstates[i],steps) #,hidden_states_d[i],cell_states_d[i])

                d_actions.append(action)
              for i in range(len(rstates)):
                action = agent_r.policy(rstates[i],steps) #,hidden_states_r[i],cell_states_r[i])

                r_actions.append(action)
                ########################################################################
              actions = d_actions + r_actions
              actions1 = []

              
              new_obs, rewards, done, _ = env.step(actions)   #new_state is also 0 or 1
              #print("step",step)
              #print("actions",actions)
              for item in actions:
                actions1.append(item.data.cpu())
              #print(actions1)  
              #for agent in env.world.agents:
              #  print(agent.name,agent.state.p_pos, end =" ")
              #print()  
              ep_actions.append(actions1)            
              d_rewards = rewards[:-2]
              r_rewards = rewards[-2:]

              
              ######### PROCESSING NEW STATE  ######################
              drone_pos_new = []
              drone_signals_new = []
              drone_warnings_new = []
              ranger_pos_new = []
              animal_densities_new = []

            
              ######### PROCESSING STATE  ######################
              for drone in env.world.drones:
                drone_pos_new.append(drone.state.p_pos)
                drone_signals_new.append(drone.state.signal)
                drone_warnings_new.append(drone.state.warn)
                animal_densities_new.append(env.world.pfr_temp[int(drone.state.p_pos[0])][int(drone.state.p_pos[1])])
              for ranger in env.world.rangers:
                ranger_pos_new.append(ranger.state.p_pos)
                poacher_pos_new = env.world.poachers[0].state.p_pos
                animal_densities_new.append(env.world.pfr_temp[int(ranger.state.p_pos[0])][int(ranger.state.p_pos[1])]) 

              
              new_states = process_states(drone_pos_new,ranger_pos_new,drone_signals_new,drone_warnings_new,animal_densities_new,new_obs[:-2],env)
              new_dstates = new_states[:-2]
              new_rstates = new_states[-2:]
            
              ######################################################

              #////////////////////////////////////////////////#

              


          
              
              #print("before",dstates[0][:,:,6])
              dstates = new_dstates  #MAKE SURE THIS WORKS PROPERLY
              #print("after",dstates[0][:,:,6])
              rstates = new_rstates  
              episode_reward_drone += sum(d_rewards)
              episode_reward_ranger += sum(r_rewards)
              #print("done?",done[0])
  
              if done[0] or step == 99:
                abc = 9
                #sys.stdout.write("episode: {},drone reward sum : {}, ranger_reward sum: {}, episode time: {}, inits: {}, poacher_init: {} \n".format(episode, np.round(episode_reward_drone, decimals=2)/3,np.round(episode_reward_ranger, decimals=2)/2 ,step+1, defender_inits, poacher_inits))#, np.mean(rewards[-10:])))
                break
              

          rewards.append(episode_reward_drone + episode_reward_ranger)
          rewards_r.append(episode_reward_drone)
          rewards_d.append(episode_reward_ranger)

          episode_reward = episode_reward_ranger + episode_reward_drone
          '''if(episode%500 == 0):
            
            plot_rewards(rewards_d, rewards_r,episode,coverages, winrates)
            visitation.append(env.world.v_counts)'''
          

          '''data_tuple = (def_allocation_counts.copy(), drone_inits, ranger_inits, episode_reward_drone, episode_reward_ranger, poacher_inits)
          rewards_dataset.append(data_tuple)
          if(episode%50 == 0):
            print(data_tuple)
      with open("dataset.txt", "wb") as fp:
        pickle.dump(rewards_dataset, fp) '''  
      #print("Reward:",episode_reward)  
      return episode_reward, env.world.v_counts, env.world.poacher_counts, ep_actions










def main():

      env = make_env.make_env("simple_tag")
      visitation = []
      winrates = []
      agent_d = RLAgent(
                      name = "drone",
                      num = 3,
                      action_space_size = 15,
                      channels = 8, #20
                      discount=0.99)


      agent_r = RLAgent(
                      name = "ranger",
                      num = 2,
                      action_space_size = 5,
                      channels = 8,
                      discount=0.99)
      batch_size = 32
      rewards = []
      rewards_r = []
      rewards_d = []
      avg_rewards = []
      coverages = []
      steps = 0
      winrate = 0
      #init0 = np.random.choice(env.world.gridsize, 5, replace = False )
      #init1 = np.random.choice(env.world.gridsize, 5, replace = False )
      #poacher_inits = [np.random.randint(0,env.world.gridsize, size = env.world.dim_p)]
      for episode in range(90000):
          
          #print("episode",episode)
          inits = []
          
          init0 = np.random.choice(env.world.gridsize, 5, replace = False )
          init1 = np.random.choice(env.world.gridsize, 5, replace = False )
          for i in range(5):
            inits.append(np.array([init0[i], init1[i]]))
          drone_inits = inits[:-2]
          ranger_inits = inits[-2:]  


        
          poacher_inits = [np.random.randint(0,env.world.gridsize, size = env.world.dim_p)]

          
          coverages.append(env.world.v_counts.copy())
          if(episode%100 == 0 and episode > 0):
              winrates.append(winrate)
              winrate = 0
      
          obs = env.reset(drone_inits, ranger_inits, poacher_inits)  #returns 0 or 1

          
          drone_pos_old = []
          drone_signals_old = []
          drone_warnings_old = []
          ranger_pos_old = []
          animal_densities_old = []
          

        
          ######### PROCESSING STATE  ######################
          for drone in env.world.drones:
          
            drone_pos_old.append(drone.state.p_pos)
            drone_signals_old.append(drone.state.signal)
            drone_warnings_old.append(drone.state.warn)
            animal_densities_old.append(env.world.pfr_temp[int(drone.state.p_pos[0])][int(drone.state.p_pos[1])])
          for ranger in env.world.rangers:
            ranger_pos_old.append(ranger.state.p_pos)
            poacher_pos_old = env.world.poachers[0].state.p_pos
            animal_densities_old.append(env.world.pfr_temp[int(ranger.state.p_pos[0])][int(ranger.state.p_pos[1])]) 

          
          states = process_states(drone_pos_old,ranger_pos_old,drone_signals_old,drone_warnings_old,animal_densities_old,obs[:-2],env)
          dstates = states[:-2]
          rstates = states[-2:]

          ######################################################



          
          episode_reward_drone = 0
          episode_reward_ranger = 0

          agentd_rewards = [[],[],[]]
          agentd_logprobs = [[],[],[]]
          agentd_values = [[],[],[]]
          agentr_rewards = [[],[]]
          agentr_logprobs = [[],[]]
          agentr_values = [[],[]]
          entropyd = 0
          entropyr = 0
          
          for step in range(100):
              steps = steps + 1
              d_actions = []
              r_actions = []
              actions = []
              d_rewards = []
              r_rewards = []
              for i in range(len(dstates)):
           
                #########################  new  #################################
                value, policy_dist = agent_d.select_action(dstates[i]) 
                #value = value.detach().numpy()#[0,0]
               
                #dist = policy_dist.detach().numpy()
                dist = Categorical(probs=policy_dist)     
                action = dist.sample()
                action = action.detach().data.numpy()
                #action = np.random.choice(15, p=np.squeeze(dist))
                log_prob = torch.log(policy_dist.squeeze(0)[action])
                entropy = 1#-np.sum(np.mean(dist) * np.log(dist))
                agentd_values[i].append(value)
                agentd_logprobs[i].append(log_prob)
                entropyd += entropy
                d_actions.append(action)
              for i in range(len(rstates)):
                value, policy_dist = agent_r.select_action(rstates[i]) 
                #value = value.detach().numpy()#[0,0]
                dist = Categorical(probs=policy_dist)     
                action = dist.sample()
                action = action.detach().data.numpy()
                log_prob = torch.log(policy_dist.squeeze(0)[action])
                entropy = 1#-np.sum(np.mean(dist) * np.log(dist))
                agentr_values[i].append(value)
                agentr_logprobs[i].append(log_prob)
                entropyr += entropy
                r_actions.append(action)
                ########################################################################
              actions = d_actions + r_actions


              
              new_obs, rewards, done, _ = env.step(actions)   #new_state is also 0 or 1
              
              d_rewards = rewards[:-2]
              for i in range(len(d_rewards)):
                  agentd_rewards[i].append(d_rewards[i])

              r_rewards = rewards[-2:]
              for i in range(len(r_rewards)):
                  agentr_rewards[i].append(r_rewards[i])             


              
              ######### PROCESSING NEW STATE  ######################
              drone_pos_new = []
              drone_signals_new = []
              drone_warnings_new = []
              ranger_pos_new = []
              animal_densities_new = []

            
              ######### PROCESSING STATE  ######################
              for drone in env.world.drones:
                drone_pos_new.append(drone.state.p_pos)
                drone_signals_new.append(drone.state.signal)
                drone_warnings_new.append(drone.state.warn)
                animal_densities_new.append(env.world.pfr_temp[int(drone.state.p_pos[0])][int(drone.state.p_pos[1])])
              for ranger in env.world.rangers:
                ranger_pos_new.append(ranger.state.p_pos)
                poacher_pos_new = env.world.poachers[0].state.p_pos
                animal_densities_new.append(env.world.pfr_temp[int(ranger.state.p_pos[0])][int(ranger.state.p_pos[1])]) 

              
              new_states = process_states(drone_pos_new,ranger_pos_new,drone_signals_new,drone_warnings_new,animal_densities_new,new_obs[:-2],env)
              new_dstates = new_states[:-2]
              new_rstates = new_states[-2:]
            
              ######################################################

              #////////////////////////////////////////////////#

              

              #/////////////////////////////////////////////////#

                  
          
              
              #print("before",dstates[0][:,:,6])
              dstates = new_dstates  #MAKE SURE THIS WORKS PROPERLY
              #print("after",dstates[0][:,:,6])
              rstates = new_rstates  
              episode_reward_drone += sum(d_rewards)
              episode_reward_ranger += sum(r_rewards)
              #print("done?",done[0])
              
              if(done[0] or step == 99):
                  agent_d.train_network(agentd_rewards, agentd_values, agentd_logprobs, entropyd)
                  agent_r.train_network(agentr_rewards, agentr_values, agentr_logprobs, entropyr)

              
              
              
              if(done[0] and step < 30):
                winrate = winrate + 1     
              if(done[0] or step == 99):
               
                
                if(episode%500==0): 
                      plt.xlabel('Attacker counts')
                      plt.imshow(env.world.poacher_counts)
                      plt.savefig('a2c2_data/attacker_counts_dqn_ep' + str(episode) + '.png')
                      plt.close()
                      plt.xlabel('Attacker probs')
                      plt.imshow(env.world.action_probs_display)
                      plt.savefig('a2c2_data/attacker_probs_dqn_ep' + str(episode) + '.png')
                      plt.close()

                if(episode%1000==0):
                  abc = 0
                  abc = abc+1#sys.stdout.write("episode: {},drone reward sum : {}, ranger_reward sum: {}, episode time: {} \n".format(episode, np.round(episode_reward_drone, decimals=2)/3,np.round(episode_reward_ranger, decimals=2)/2 ,step+1))#, np.mean(rewards[-10:])))
                break
              

          rewards.append(episode_reward_drone + episode_reward_ranger)
          rewards_r.append(episode_reward_drone)
          rewards_d.append(episode_reward_ranger)
        
          if(episode%2000 == 0):
                np.save('a2c2_data/winrate_dqn_ep' + str(episode),np.array(winrates))
                np.save('a2c2_data/rewards_dqn_ep' + str(episode),np.array(rewards))
                np.save('a2c2_data/rewards_dqn_ep' + str(episode),np.array(rewards_r))
                np.save('a2c2_data/rewards_dqn_ep' + str(episode),np.array(rewards_d))
          
          if(episode%500 == 0):
            
            plot_rewards(rewards_d, rewards_r,episode,coverages, winrates)
            visitation.append(env.world.v_counts)

          if(episode%5000 == 0 and episode > 0):
            torch.save(agent_d.network.state_dict(),"models/drone_a2c2" + str(episode) + ".pth")
            torch.save(agent_r.network.state_dict(),"models/ranger_a2c2" + str(episode) + ".pth")
            #if(episode > 0):
            #  evaluate()


            


if __name__ == '__main__':
    main()
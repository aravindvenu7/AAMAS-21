import gym   #might run into version issue
import pickle
import make_env
from maddpg import MADDPG
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
from collections import deque
import random
from scipy.stats import rankdata
import itertools, math
from torch.distributions import Categorical, Multinomial
seed = 0
from storage_maddpg import ReplayBuffer
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
print("Seed 0")   
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    plt.savefig('maddpg_data/rewards_dqn_ep' + str(episode) + '.png')
    plt.close()

    plt.plot(winrates)
    plt.xlabel('Episode*100')
    plt.ylabel('Winrate')
    plt.savefig('maddpg_data/winrates_dqn_ep' + str(episode) + '.png')
    plt.close()

    plt.xlabel('dqn_data/Visitation counts')
    plt.imshow(counts[-1])
    plt.savefig('maddpg_data/v_counts_dqn_ep' + str(episode) + '.png')
    plt.close()

    plt.xlabel('Attacker probability acc to coverage')
    plt.imshow(coverage)
    plt.savefig('maddpg_data/attacker_prob_dqn_ep' + str(episode) + '.png')
    plt.close()


 
    plt.xlabel('Coverage - Prev. Coverage')
    x = abs(coverage - prev_coverage)
    plt.imshow(x)
    plt.savefig('maddpg_data/coverage_diff_dqn_ep' + str(episode) + '.png')
    plt.close()


    if(episode > 0 ):
      y = abs(counts[-1] - counts[-100])
      plt.xlabel('Count - Prev. Count')
      plt.imshow(y)
      plt.savefig('maddpg_data/counts_diff_dqn_ep' + str(episode) + '.png')
      plt.close()



def load_models():

      maddpg = MADDPG()

     
      
      
      maddpg.agents[0].policy.load_state_dict(torch.load("maddpg_models/drone_policy500.pth"))
      maddpg.agents[0].target_policy.load_state_dict(torch.load("models/drone_tpolicy500.pth"))      
      maddpg.agents[0].critic.load_state_dict(torch.load("maddpg_models/drone_critic500.pth"))      
      maddpg.agents[0].target_critic.load_state_dict(torch.load("maddpg_models/drone_tcritic500.pth"))
        
      maddpg.agents[1].policy.load_state_dict(torch.load("maddpg_models/ranger_policy500.pth"))
      maddpg.agents[1].target_policy.load_state_dict(torch.load("maddpg_models/ranger_tpolicy500.pth"))      
      maddpg.agents[1].critic.load_state_dict(torch.load("maddpg_models/ranger_critic500.pth"))      
      maddpg.agents[1].target_critic.load_state_dict(torch.load("maddpg_models/ranger_tcritic500.pth"))

      return maddpg   



def evaluate():


      maddpg = load_models()      
      env = make_env.make_env("simple_tag")
      visitation = []
      winrates = []
      def_allocation_counts = np.full((env.world.gridsize, env.world.gridsize), 1)
      rewards = []
      rewards_r = []
      rewards_d = []
      batch_size = 32
      coverages = []
      steps = 500000
      winrate = 0
      ep_actions = []
      for episode in range(500):
          
      
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
          if(episode%1000 == 0):
                print("episode", episode)


          
          for step in range(100):
              steps = steps + 1
              d_actions = []
              r_actions = []
              actions = []
              d_rewards = []
              r_rewards = []
              for i in range(len(dstates)):
                #########################  new  #################################
                
                action = maddpg.agents[0].step(Variable(torch.from_numpy(dstates[i]).float().unsqueeze(0)).to(device)) #,hidden_states_d[i],cell_states_d[i])
                #hidden_states_d[i] = h.to(device)
                #cell_states_d[i] =   c.to(device)
                d_actions.append(action)
              for i in range(len(rstates)):
                action = maddpg.agents[1].step(Variable(torch.from_numpy(rstates[i]).float().unsqueeze(0)).to(device)) #,hidden_states_r[i],cell_states_r[i])
                #hidden_states_r[i] = h.to(device)
                #cell_states_r[i] = c.to(device)
                r_actions.append(action)
                ########################################################################
              actions = d_actions + r_actions
              real_actions = []
              for action in actions:
                action = np.array(action.cpu().detach().clone())
                action_max = np.array([np.argmax(action[i]) for i in range(action.shape[0])])
                real_actions.append(action_max)         
              
              new_obs, rewards, done, _ = env.step(real_actions)   #new_state is also 0 or 1
              
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

              

              #/////////////////////////////////////////////////#

              
              dones = []
              states_store = []
              new_states_store = []
              for item in states:
                states_store.append(item.reshape((8*15*15))) 
              for item in new_states:
                new_states_store.append(item.reshape((8*15*15)))                 
              
              for i in range(len(actions)):
                  dones.append(done[0])

              actions_store = []
              for item in actions:
                item = item.cpu().numpy()
                actions_store.append(item)
              rewards = [rewards]
              rewards = np.array(rewards)
              dones = np.array(dones).reshape((1,5))
                  
          
              
              #print("before",dstates[0][:,:,6])
              dstates = new_dstates  #MAKE SURE THIS WORKS PROPERLY
              #print("after",dstates[0][:,:,6])
              rstates = new_rstates  
              episode_reward_drone += sum(d_rewards)
              episode_reward_ranger += sum(r_rewards)
              rewards = rewards.tolist()
              #print("done?",done[0])
              

              if(done[0] and step < 30):
                winrate = winrate + 1     
              if done[0] or step == 99:
               

                if(episode%1000==0):
                  abc = 0
                  abc = abc+1#sys.stdout.write("episode: {},drone reward sum : {}, ranger_reward sum: {}, episode time: {} \n".format(episode, np.round(episode_reward_drone, decimals=2)/3,np.round(episode_reward_ranger, decimals=2)/2 ,step+1))#, np.mean(rewards[-10:])))
                break
              

          rewards.append(episode_reward_drone + episode_reward_ranger)
          rewards_r.append(episode_reward_drone)
          rewards_d.append(episode_reward_ranger)
        
      np.save("rewards_maddpg.npy", np.array(rewards))
      return rewards










def main():

      env = make_env.make_env("simple_tag")
      visitation = []
      winrates = []
      maddpg = MADDPG()
      print("making replay buffer")
      replay_buffer = ReplayBuffer(int(4e4), 5,
                                 [15*15*8, 15*15*8,15*15*8, 15*15*8,15*15*8],
                                 [15,15,15,5,5])
      print("made replay buffer")
      batch_size = 32
      rewards = []
      rewards_r = []
      rewards_d = []
      avg_rewards = []
      coverages = []
      steps = 0
      winrate = 0
      for episode in range(500):
          
      
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
          if(episode%500 == 0):
                print("episode", episode)


          
          for step in range(100):
              steps = steps + 1
              d_actions = []
              r_actions = []
              actions = []
              d_rewards = []
              r_rewards = []
              for i in range(len(dstates)):
                #########################  new  #################################
                
                action = maddpg.agents[0].step(Variable(torch.from_numpy(dstates[i]).float().unsqueeze(0)).to(device)) #,hidden_states_d[i],cell_states_d[i])
                #hidden_states_d[i] = h.to(device)
                #cell_states_d[i] =   c.to(device)
                d_actions.append(action)
              for i in range(len(rstates)):
                action = maddpg.agents[1].step(Variable(torch.from_numpy(rstates[i]).float().unsqueeze(0)).to(device)) #,hidden_states_r[i],cell_states_r[i])
                #hidden_states_r[i] = h.to(device)
                #cell_states_r[i] = c.to(device)
                r_actions.append(action)
                ########################################################################
              actions = d_actions + r_actions
              real_actions = []
              for action in actions:
                action = np.array(action.cpu().detach().clone())
                action_max = np.array([np.argmax(action[i]) for i in range(action.shape[0])])
                real_actions.append(action_max)         
              
              new_obs, rewards, done, _ = env.step(real_actions)   #new_state is also 0 or 1
              
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

              

              #/////////////////////////////////////////////////#

              
              dones = []
              states_store = []
              new_states_store = []
              for item in states:
                states_store.append(item.reshape((8*15*15))) 
              for item in new_states:
                new_states_store.append(item.reshape((8*15*15)))                 
              
              for i in range(len(actions)):
                  dones.append(done[0])

              actions_store = []
              for item in actions:
                item = item.cpu().numpy()
                actions_store.append(item)
              rewards = [rewards]
              rewards = np.array(rewards)
              dones = np.array(dones).reshape((1,5))

              replay_buffer.push(np.array(states_store).reshape((1,5,1800)), actions_store, rewards, np.array(new_states_store).reshape((1,5,1800)), dones)  #actions have to be one-hot
       


              
              for a_i in range(2):
                if (len(replay_buffer) >= batch_size*6):
                        sample = replay_buffer.sample(batch_size,to_gpu=False)                   
                        maddpg.update(sample, a_i)
                        maddpg.update_all_targets()
                  
          
              
              #print("before",dstates[0][:,:,6])
              dstates = new_dstates  #MAKE SURE THIS WORKS PROPERLY
              #print("after",dstates[0][:,:,6])
              rstates = new_rstates  
              episode_reward_drone += sum(d_rewards)
              episode_reward_ranger += sum(r_rewards)
              rewards = rewards.tolist()
              #print("done?",done[0])
              

              if(done[0] and step < 30):
                winrate = winrate + 1     
              if done[0] or step == 99:
               
                
                if(episode%500==0): 
                      plt.xlabel('Attacker counts')
                      plt.imshow(env.world.poacher_counts)
                      plt.savefig('maddpg_data/attacker_counts_dqn_ep' + str(episode) + '.png')
                      plt.close()
                      plt.xlabel('Attacker probs')
                      plt.imshow(env.world.action_probs_display)
                      plt.savefig('maddpg_data/attacker_probs_dqn_ep' + str(episode) + '.png')
                      plt.close()

                if(episode%1000==0):
                  abc = 0
                  abc = abc+1#sys.stdout.write("episode: {},drone reward sum : {}, ranger_reward sum: {}, episode time: {} \n".format(episode, np.round(episode_reward_drone, decimals=2)/3,np.round(episode_reward_ranger, decimals=2)/2 ,step+1))#, np.mean(rewards[-10:])))
                break
              

          rewards.append(episode_reward_drone + episode_reward_ranger)
          rewards_r.append(episode_reward_drone)
          rewards_d.append(episode_reward_ranger)
        
          if(episode%100 == 0):
                np.save('maddpg_data/winrate_dqn_ep' + str(episode),np.array(winrates))
                np.save('maddpg_data/rewards_dqn_ep' + str(episode),np.array(rewards))
          
          if(episode%500 == 0):
            
            plot_rewards(rewards_d, rewards_r,episode,coverages, winrates)
            visitation.append(env.world.v_counts)

          if(episode%1000 == 0):
            torch.save(maddpg.agents[0].policy.state_dict(),"maddpg_models/drone_policy" + str(episode) + ".pth")
            torch.save(maddpg.agents[0].critic.state_dict(),"maddpg_models/drone_critic" + str(episode) + ".pth")
            torch.save(maddpg.agents[0].target_policy.state_dict(),"maddpg_models/drone_tpolicy" + str(episode) + ".pth")
            torch.save(maddpg.agents[0].target_critic.state_dict(),"maddpg_models/drone_tcritic" + str(episode) + ".pth")
            torch.save(maddpg.agents[1].policy.state_dict(),"maddpg_models/ranger_policy" + str(episode) + ".pth")
            torch.save(maddpg.agents[1].critic.state_dict(),"maddpg_models/ranger_critic" + str(episode) + ".pth")
            torch.save(maddpg.agents[1].target_policy.state_dict(),"maddpg_models/ranger_tpolicy" + str(episode) + ".pth")
            torch.save(maddpg.agents[1].target_critic.state_dict(),"maddpg_models/ranger_tcritic" + str(episode) + ".pth")


            


if __name__ == '__main__':
    main()
    evaluate()
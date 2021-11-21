
import gym   #might run into version issue
import make_env
from agent_nfsp import RLAgent
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


def process_states(dronepos,rangerpos,dronesig,dronewarn,densities,obs):
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
    state = np.zeros((15,15,7))
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
    plt.savefig('nfsp_data/rewards_nfsp_ep' + str(episode) + '.png')
    plt.close()

    plt.plot(winrates)
    plt.xlabel('Episode*100')
    plt.ylabel('Winrate')
    plt.savefig('nfsp_data/winrates_nfsp_ep' + str(episode) + '.png')
    plt.close()

    plt.xlabel('nfsp_data/Visitation counts')
    plt.imshow(counts[-1])
    plt.savefig('nfsp_data/v_counts_nfsp_ep' + str(episode) + '.png')
    plt.close()

    plt.xlabel('Attacker probability acc to coverage')
    plt.imshow(coverage)
    plt.savefig('nfsp_data/attacker_prob_nfsp_ep' + str(episode) + '.png')
    plt.close()


 
    plt.xlabel('Coverage - Prev. Coverage')
    x = abs(coverage - prev_coverage)
    plt.imshow(x)
    plt.savefig('nfsp_data/coverage_diff_nfsp_ep' + str(episode) + '.png')
    plt.close()


    if(episode > 0 ):
      y = abs(counts[-1] - counts[-100])
      plt.xlabel('Count - Prev. Count')
      plt.imshow(y)
      plt.savefig('nfsp_data/counts_diff_nfsp_ep' + str(episode) + '.png')
      plt.close()



def main():
        env = make_env.make_env("simple_tag")
        visitation = []
        winrates = []
        agent_d = RLAgent(
                        name = "drone",
                        action_space_size = 15,
                        channels = 7,
                        target_update_freq= 20,  #20
                        discount=0.99,
                        batch_size=32,
                        max_explore=1,
                        min_explore=0.05,
                        anneal_rate=(9 / 10000),
                        replay_memory_size=320*6,
                        sl_memory_size = 100000,  #320x120 
                        replay_start_size=32*6)


        agent_r = RLAgent(
                        name = "ranger",
                        action_space_size = 5,
                        channels = 7,
                        target_update_freq= 50,  #50
                        discount=0.99,
                        batch_size=32,
                        max_explore=1,
                        min_explore=0.05,
                        anneal_rate=(9 / 10000),
                        replay_memory_size=320*2,
                        sl_memory_size = 100000,  #32x40 
                        replay_start_size=32*2)
        batch_size = 32
        rewards = []
        rewards_r = []
        rewards_d = []
        avg_rewards = []
        coverages = []
        steps = 0
        winrate = 0
        greedy_policy = False
        avg_policy = False
        for episode in range(100000):
            
            '''if(random.random() > 0.1):
            greedy_policy = False
            avg_policy = True
            else:
            greedy_policy = True
            avg_policy = False ''' 
            
        
            coverages.append(env.world.v_counts.copy())

            if(episode%100 == 0 and episode > 0):
                winrates.append(winrate)
                winrate = 0
            obs = env.reset()  #returns 0 or 1
            
            
            drone_pos_old = []
            drone_signals_old = []
            drone_warnings_old = []
            ranger_pos_old = []
            animal_densities_old = []
            
            #print("start")

            '''for agent in env.world.agents:
                print(agent.name,agent.state.p_pos, end =" ")
            print()  '''
        
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

            
            states = process_states(drone_pos_old,ranger_pos_old,drone_signals_old,drone_warnings_old,animal_densities_old,obs[:-2])
            dstates = states[:-2]
            rstates = states[-2:]

            ######################################################



            
            episode_reward_drone = 0
            episode_reward_ranger = 0


            
            for step in range(100):
                
                steps = steps + 1
                sl_d_actions = []
                is_best_d = [False]*6
                is_best_r = [False]*2

                sl_r_actions = []
                d_actions = []
                r_actions = []
                actions = []
                d_rewards = []
                r_rewards = []
                for i in range(len(dstates)):

                    '''if(greedy_policy and not avg_policy):
                    action = agent_d.greedy_policy(dstates[i],steps)
                    else:
                    action = agent_d.avg_policy(dstates[i],steps)'''
            
                    action, is_best_d[i] = agent_d.policy(dstates[i],steps, is_best_d[i], episode) 

                    d_actions.append(action)
                for i in range(len(rstates)):

                    '''if(greedy_policy and not avg_policy):
                    action = agent_r.greedy_policy(rstates[i],steps)
                    else:
                    
                    action = agent_r.avg_policy(rstates[i],steps)'''
                    action, is_best_r[i] = agent_r.policy(rstates[i],steps, is_best_r[i], episode) 

                    r_actions.append(action)
                
                actions = d_actions + r_actions
                '''if(episode > 400):
                    if(episode%100==0):
                        print(actions)'''

                
                new_obs, rewards, done, _ = env.step(actions)   #new_state is also 0 or 1
                
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

                
                new_states = process_states(drone_pos_new,ranger_pos_new,drone_signals_new,drone_warnings_new,animal_densities_new,new_obs[:-2])
                new_dstates = new_states[:-2]
                new_rstates = new_states[-2:]
            
                ######################################################

                #////////////////////////////////////////////////#

                

                #/////////////////////////////////////////////////#

                for i in range(len(env.world.drones)):
                
                
                    experience_d = {
                                "state": dstates[i],
                                "action": d_actions[i],
                                "reward": d_rewards[i],
                                "next_state": new_dstates[i],
                                "done": done[0]
                                }
                    if(is_best_d[i] == True):  
                        agent_d.slmemory.add(dstates[i], d_actions[i])
                    agent_d.rlmemory.add(experience_d)


                for i in range(len(env.world.rangers)):

                    experience_r = {
                                "state": rstates[i],
                                "action": r_actions[i],
                                "reward": r_rewards[i],
                                "next_state": new_rstates[i],
                                "done": done[0]
                                }
                    if(is_best_r[i] == True):   
                        agent_r.slmemory.add(rstates[i],r_actions[i])

                    agent_r.rlmemory.add(experience_r)          

                
                if len(agent_d.rlmemory) > batch_size*6:
                    agent_d.train_rl_network(steps)     #THIS ALSO UPDATES TARGET NETWORK
                    
                    if(len(agent_d.slmemory) > batch_size*6):
                        agent_d.train_sl_network()
                
                if len(agent_r.rlmemory) > batch_size*2:
                    agent_r.train_rl_network(steps)     #THIS ALSO UPDATES TARGET NETWORK
                
                    if(len(agent_r.slmemory) > batch_size*2):
                        agent_r.train_sl_network()
                    
            
                
                
                dstates = new_dstates  #MAKE SURE THIS WORKS PROPERLY
          
                rstates = new_rstates  
                episode_reward_drone += sum(d_rewards)
                episode_reward_ranger += sum(r_rewards)
              
                '''if(episode%100==0):
                    for agent in env.world.agents:
                        print(env.world.detected_first_time,agent.name,agent.state.p_pos, end =" ")
                    print() '''
                if(done[0] and step < 30):
                    winrate = winrate + 1  
                if done[0] or step == 99:
                    
                    if(episode%500==0): 
                        plt.xlabel('Attacker counts')
                        plt.imshow(env.world.poacher_counts)
                        plt.savefig('nfsp_data/attacker_counts_nfsp_ep' + str(episode) + '.png')
                        plt.close()
                        plt.xlabel('Attacker probs')
                        plt.imshow(env.world.action_probs_display)
                        plt.savefig('nfsp_data/attacker_probs_nfsp_ep' + str(episode) + '.png')
                        plt.close()
                    if(episode%500==0 and episode > 0):
                        agent_d.plot_gradients(agent_d.name, episode)
                        agent_r.plot_gradients(agent_r.name, episode)
                        abc = 0
                        abc = abc + 1#sys.stdout.write("episode: {},drone reward sum : {}, ranger_reward sum: {}, episode time: {} \n".format(episode, np.round(episode_reward_drone, decimals=2)/3,np.round(episode_reward_ranger, decimals=2)/2 ,step+1))#, np.mean(rewards[-10:])))
                    break
                
                

            rewards.append(episode_reward_drone + episode_reward_ranger)
            rewards_r.append(episode_reward_drone)
            rewards_d.append(episode_reward_ranger)
        
            if(episode%2000 == 0):
                np.save('nfsp_data/winrate_nfsp_ep' + str(episode),np.array(winrates))
            if(episode%500 == 0):
            
                plot_rewards(rewards_d, rewards_r,episode,coverages, winrates)
                visitation.append(env.world.v_counts)

if __name__ == '__main__':
    main()



        #Add better strcture to code - kind of done
        #Remove all printed statements and save them instead - done
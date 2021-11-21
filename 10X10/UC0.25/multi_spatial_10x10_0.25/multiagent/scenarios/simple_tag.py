# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import math
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario
from scipy.stats import rankdata

class Scenario(BaseScenario):





    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 0
        num_drones = 3
        num_rangers = 2
        num_poachers = 2




        num_agents = num_drones + num_rangers + num_poachers
        num_landmarks = 0
        
        
        # add agents
        world.drones = [Agent() for i in range(num_drones)]
        world.poachers = [Agent() for i in range(num_poachers)]
        world.rangers = [Agent() for i in range(num_rangers)]
        


        
        ######################################### DEFINE DRONE, ATTACKER AND RANGER HERE #######################################################################
        i = 0
        for drone in world.drones:
          
          drone.name = "drone"
          drone.color = np.array([0.0,0.0,255.0])           #blue
          drone.silent = True
          drone.id = i
          i = i+1
          

        
        for ranger in world.rangers:
          ranger.name = "ranger" 
          ranger.color = np.array([0.0,255.0,0.0])           #green
          ranger.silent = True   

        i = 0
        for poacher in world.poachers:
          poacher.name = "poacher"
          poacher.color = np.array([255.0,0.0,0.0])         #red
          poacher.adversary = True        
          poacher.action_callback = 1 
          poacher.id = i 
          i = i+1

        world.agents = world.drones +  world.rangers + world.poachers
        world.defenders = world.drones + world.rangers
        
        for p in range(len(world.drones)):
          world.drone_inits.append(np.random.randint(10, size = world.dim_p))
        for q in range(len(world.rangers)):
          world.ranger_inits.append(np.random.randint(10, size = world.dim_p))
          
        world.drone_inits = [np.array([5,5]),np.array([7,5]), np.array([9,5]),np.array([5,9]), np.array([7,9]), np.array([9,9])]
        #world.drone_inits = [np.array([3,3]),np.array([7,3]), np.array([11,3]),np.array([3,11]), np.array([7,11]), np.array([11,11])]  
        world.ranger_inits = [np.array([6,7]), np.array([8,7])]
        world.poacher_inits =  [np.array([0,1]), np.array([1,0])]
        #print("drone inits",world.drone_inits)
        #print("ranger inits",world.ranger_inits)
        #print("poacher inits",world.poacher_inits)
        #########################################################################################################################################################




        self.reset_world(world, world.drone_inits, world.ranger_inits, world.poacher_inits)
        return world


    def reset_world(self, world, drone_inits, ranger_inits, poacher_inits):
        
        
        
        world.detected_first_time = False
        world.time = 0
        world.time_of_detection = 0
        
 
        '''world.animal_densities = world.animal_densities_initial.copy()
        world.poacher_policy = np.zeros((world.gridsize,world.gridsize))'''
 
        





        i = 0
        for agent in world.drones:
          
            agent.state.p_pos = drone_inits[i]#np.random.randint(2,15, size = world.dim_p)
            agent.state.warn = False
            agent.state.signal = False
            i = i+1
        i = 0
        for agent in world.rangers:    
          
            
            agent.state.p_pos = ranger_inits[i]#np.random.randint(2,15, size = world.dim_p)
            agent.state.warn = False
            agent.state.signal = False
            i = i+1
        i = 0
        for agent in world.poachers:
            
            agent.state.p_pos = poacher_inits[i]#np.array([11,7])
            agent.is_caught = False
            agent.is_flee = False
            agent.is_detected = False
            agent.reward = 0
            agent.reached_target = False
            agent.reached_int_target = False
            agent.left_park = False
            agent.targets = []
            i = i+1
        i = 0    


        for i in range (world.gridsize):
          for j in range (world.gridsize):
            min_d = 1000
            #sumd = 0
            #avd = 0
            for agent in world.defenders:
              d = abs(agent.state.p_pos[0] - i) + abs(agent.state.p_pos[1] - j)
              if(d < min_d):
                min_d = d
              #sumd = sumd + d
            #avd = sumd / len(world.defenders)  
            world.distances[i][j] = min_d

        dist = world.distances.copy().reshape((100))
        
        dr = rankdata(dist,method = 'ordinal') / 100.0
        
        counts = world.v_counts.copy().reshape((100))
        countrank = rankdata(counts,method = 'ordinal') 
        countrank = (100 - countrank) / 100.0


        
        
        
        ad = world.animal_densities.reshape((100))
        
        
        adr1 = 0.5*ad + 0.5*dr   #0.5*(countrank) # Combination of animal density and distance score wrt initial positions of defender agents.

        adr1 = rankdata(adr1,method = 'ordinal') / 100.0
        adr1 = adr1.reshape((10,10))
        
        if(np.amax(world.poacher_policy_complete) == 0):
          world.poacher_policy_complete = adr1.copy()
        else:
      
          world.poacher_policy_complete = 0.9*world.poacher_policy_complete + 0.1*(adr1.copy())  #Update rule. ie poacher trajectory changes as the inits of defender change
        
        world.poacher_policy = world.poacher_policy_complete.copy()
        world.action_probs_display = world.poacher_policy_complete.copy()

        for agent in world.poachers:
          ranks = rankdata(world.poacher_policy.copy().reshape((world.gridsize*world.gridsize)),method = 'ordinal').reshape((world.gridsize,world.gridsize))
          for i in range(world.gridsize):
           for j in range(world.gridsize):
            if(abs(agent.state.p_pos[0] - i) > 1 or abs(agent.state.p_pos[1] - j) > 1):
              ranks[i][j] = 0 
          i, j = np.where( ranks == np.amax(ranks))  
          agent.target[0] = i
          agent.target[1] = j
          agent.reward = agent.reward + np.amax(world.animal_densities)    
          world.poacher_policy[int(agent.target[0])][int(agent.target[1])] = 0.0
        world.visits = np.zeros((world.gridsize,world.gridsize))
        world.poacher_counts = np.zeros((world.gridsize,world.gridsize))   

        



    def benchmark_data(self, agent, world):          
       
            detections = 0
       
                           
            if self.is_detected(agent,world,0):
                    detections += 1
            return detections
     

    #returns true if agent1's position is same as agent2's position 

    
    def is_detected(self, agent1, world, key):     #LIKE A HELPER FUNCTION FOR THE OBSERVATION FUNCTION
        distsx = []
        delta_pos = 0
        uc = np.random.rand(1)
        for agentp in world.poachers:
          
          delta_pos = agent1.state.p_pos - agentp.state.p_pos
          if(np.sqrt(np.sum(np.square(delta_pos))) == 0):
            if(uc>0.25):
             agentp.is_detected = True
            if(agent1.action.warn):
              if(uc>0.25):
               agentp.is_flee  = True

          distsx.append(np.sqrt(np.sum(np.square(delta_pos))))

        for distx in distsx:
          if(distx == 0):
           
           if(uc>0.25):
            if(world.detected_first_time == False):
              world.detected_first_time = True
              world.time_of_detection = world.time
            return True
        
        return False  


      
        
    def samecell(self,agent,world):
      
      sc = 0
      all_a = world.drones + world.rangers
      for a in all_a:
        delta_pos = agent.state.p_pos - a.state.p_pos
        dist = np.sum(np.square(delta_pos))
        if(dist == 0):
          sc = sc+1

      if(sc == 1):
        return False
      else:
        return True      



    def is_caught(self,world):     #LIKE A HELPER FUNCTION FOR THE OBSERVATION FUNCTION
      
      for agent2 in world.poachers:
        for agent1 in world.rangers:
                                         
          if((agent1.state.p_pos[0] - agent2.state.p_pos[0] == 0) and (agent1.state.p_pos[1] - agent2.state.p_pos[1] == 0)):
            agent2.is_caught = True
            

      '''if(world.poachers[0].is_caught or world.poachers[1].is_caught):
        return True
      else:  
        return False  '''    
        
    
    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]




    def reward(self, agent, world):           
        

      main_reward = self.agent_reward(agent,world) 
  
      return main_reward

    def agent_reward(self,agent,world):   
       rew = 0
       uc1 = np.random.rand(1)
       if(agent.name == "drone"):

          if(world.visits[int(agent.state.p_pos[0])][int(agent.state.p_pos[1])] == 0):
              rew = rew + world.pfr_temp[int(agent.state.p_pos[0])][int(agent.state.p_pos[1])]*1

          if(self.samecell(agent,world)):
              rew = rew - 2
              #print("same cell penalty drone:",agent)

          if(agent.action.warn and self.is_detected(agent,world,1)):   
                qwerty = 0
                #print("premature flee activated maybe ")


                
                                        
          '''if(self.is_detected(agent,world) and (agent.action.warn == True or agent.action.signal == True )):
              rew = rew + 2  #REWARD DETECTION
              print("true detection drone",agent)'''
          if((not self.is_detected(agent,world,0)) and (agent.action.warn == True or agent.action.signal == True )):
         
              rew = rew - 10 #PENALIZE FALSE DETECTION 
              #print("false detection drone",agent) 



       if(agent.name == "ranger"):

          '''if(world.visits[int(agent.state.p_pos[0])][int(agent.state.p_pos[1])] == 0 and not world.sig_first):
              print("moves without signal")
              rew = rew - 20'''
           
             
          if(self.samecell(agent,world) ):
              rew = rew - 2
              #print("same cell penalty ranger",agent)

              
                                    
    
       return rew

    def shared_rew(self,world):
        
      srew = 0
      self.is_caught(world)
      for p in world.poachers:
       if(p.is_flee):   
         srew = srew + 0
         #print("premature flee poacher ",p.id)

       if(not p.is_caught and not p.is_flee):  
         #print("not caught")
         srew = srew - world.pfr_temp[int(p.target[0])][int(p.target[1])]*160 #Here, add a penalty proportional to animal density of poacher's cell for each timestep poacher is not caught'''
       


       
       if(p.is_caught and not p.left_park):
           #print("caught poacher", p.id)           
           srew = srew + world.pfr_temp[int(p.target[0])][int(p.target[1])]*160*0.003 
           #print("here",p.state.p_pos)

      return srew       



       
    def observation(self,agent,world):          #FOR OBSERVATION OF DRONE
          
        if(agent.name == "drone"):
          obsr = 0
          if(self.is_detected(agent,world,0)):
            obsr = 1
          return obsr

        signals = []
        if(agent.name == "ranger"):
          for d in world.drones:

            if(d.state.signal):
              signals.append(1)
            else:
              signals.append(0)

          return signals   
           
        
    
    def episode_done(self,world):

      end_condition_escape = 0
      end_condition_caught = 0
      for agent in world.poachers:
        if(self.escaped(world, agent)):  
          agent.state.p_pos[0] = -99
          agent.state.p_pos[1] = -99
          agent.left_park = True
          end_condition_escape = end_condition_escape + 1
        else:

          if(agent.is_caught):
            end_condition_caught = end_condition_caught + 1
      if(end_condition_caught + end_condition_escape > 1):
        return True
      else:
        return False      


      

      '''if((world.poachers[0].is_caught and world.poachers[1].is_caught) or (world.poachers[0].is_flee and world.poachers[1].is_flee)
         or (world.poachers[0].is_caught and world.poachers[1].is_flee) or (world.poachers[0].is_flee and world.poachers[1].is_caught)): # and world.detected_first_time):
        print("poacher  0 caught/flee",world.poachers[0].is_caught, world.poachers[0].is_flee, end = " ")
        print("poacher  1 caught/flee",world.poachers[1].is_caught, world.poachers[1].is_flee)
        
        
        return True
      else:
        return False  

      if((world.poachers[0].is_caught and world.poachers[0].is_flee) or (world.poachers[1].is_caught and world.poachers[1].is_flee)) :
        print("ANOMALY") 
        for dr in world.drones:
          print(dr.name, dr.state.p_pos)
        for ra in world.rangers:
          print(ra.name, ra.state.p_pos)'''



    def escaped(self,world,agent):
      
      posp = agent.state.p_pos
      if(agent.is_caught):
        return False
      else:  
        if(agent.reached_target or agent.is_flee):
          if(posp[0] == 0 or posp[0] == (world.gridsize-1) or posp[1] == 0 or posp[1] == (world.gridsize-1)):
            return True
          else:
            return False  
        else:
          return False 


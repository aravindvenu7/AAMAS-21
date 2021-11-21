

import numpy as np
from scipy import signal
from scipy.stats import rankdata
# physical/external base state of all entites
class EntityState(object):
    def __init__(self):
        # physical position
        self.p_pos = None
        #self.p_pos = np.zeros(2)

# state of agents (including communication and internal/mental state)
class AgentState(EntityState):
    def __init__(self):
        super(AgentState, self).__init__()
        # communication utterance
        self.c = None
        self.signal = False
        self.warn = False

# action of the agent
class Action(object):
    def __init__(self):
        # physical action
        self.u = np.zeros(2)
        #Warning 
        self.warn = False
        #Signalling
        self.signal = False
        # communication action
        self.c = None

# properties and state of physical world entity
class Entity(object):
    def __init__(self):
        # name 
        self.name = ''
        # properties:
        
        # entity can move / be pushed
        self.movable = False
        self.size = 0.20
        # color
        self.color = None
        self.warn_color = np.array([0.0,0.0,0.0])
        self.signal_color = np.array([255.0,255.0,0.0])
       
       
        # state
        self.state = EntityState()
        
        # mass
        self.initial_mass = 1.0

    @property
    def mass(self):
        return self.initial_mass

# properties of landmark entities
class Landmark(Entity):
     def __init__(self):
        super(Landmark, self).__init__()

# properties of agent entities
class Agent(Entity):
    def __init__(self):
        super(Agent, self).__init__()
        # agents are movable by default
        self.movable = True
        # cannot send communication signals
        self.silent = True
        # cannot observe the world
        self.blind = False
        # physical motor noise amount
        self.u_noise = None
        # communication noise amount
        self.c_noise = None
        # control range
        self.u_range = 1.0          #NOT SURE WHAT THIS MEANS. MIGHT BE IMPORTANT LATER
        # state
        self.state = AgentState()
        #adversary or not
        self.adversary = False
        # action
        self.action = Action()
        # script behavior to execute
        self.action_callback = None  #  CHANGES TO BE MADE HERE TO DEFINE POLICIES FOR POACHER AND RANGER

# multi-agent world
class World(object):
    def __init__(self):
        # list of agents and entities (can change at execution-time!)
        self.episodes = 0
        self.last_flee_episode = 0
        self.last_caught_episode = 0
        
        
        self.capture_pos = None
        self.flee_pos = None
        self.drones = []
        self.poachers = []
        self.rangers = []
        self.agents = []
        self.landmarks = []
        self.defenders = []

        self.targets = []
        self.gridsize = 15
        self.poacher_policy = np.zeros((self.gridsize,self.gridsize))
        self.action_probs_display = np.zeros((self.gridsize,self.gridsize))

        self.poacher_policy_complete = np.zeros((self.gridsize,self.gridsize))
        self.drone_inits = []
        self.ranger_inits = []
        self.poacher_inits = []


        self.v_counts = np.zeros((self.gridsize,self.gridsize))
        self.poacher_counts = np.zeros((self.gridsize,self.gridsize))
        self.distances = np.zeros((self.gridsize,self.gridsize))
        self.detected_first_time = False         #Set to true the first time the drone detects the poacher
        self.time = 0   #VARIABLE TO KEEP TRACK OF TIME
        self.time_of_detection = 0  # TIME STEP WHEN POACHER IS FIRST DETECTED
        self.flee_time = 0
        # communication channel dimensionality
        self.dim_c = 0
        # position dimensionality
        self.dim_p = 2
        # color dimensionality
        self.dim_color = 3

        self.reached_target = False  #TO SIGNAL WHETHER POACHER REACHED TARGET REWARD 
        self.target = np.array([6,6]) #TARGET THAT THE POACHER HAS TO REACH

        self.premature_flee = False
        #self.saw_target = False


        '''river_dist = np.zeros((15,15))
        boundary_dist =  np.zeros((15,15))
        road_dist =  np.zeros((15,15))

        river = []
        road = []
        boundary = []

        for i in range(15):
          for j in range(15):
            if(i+j == 9):
              river_dist[i][j] = 1
              river.append((i,j))
            if(i+j == 18):
              road_dist[i][j] = 1  
              road.append((i,j))
            if(i == 0 or i == 14 or j == 0 or j == 14):
              boundary_dist[i][j] = 1
              boundary.append((i,j))  

        for i in range(15):
          for j in range(15):
            dists_r = []
            dists_ro = []
            dists_b = []
            for item in river:
              dists_r.append(abs(item[0] - i) + abs(item[1] - j))
            for item in road:
              dists_ro.append(abs(item[0] - i) + abs(item[1] - j))
            for item in boundary:
              dists_b.append(abs(item[0] - i) + abs(item[1] - j))
            river_dist[i][j] = min(dists_r)
            road_dist[i][j] = min(dists_ro)
            boundary_dist[i][j] = min(dists_b) 

        river_dist = river_dist.reshape((225))
        road_dist = road_dist.reshape((225))
        boundary_dist = boundary_dist.reshape((225))    '''       

        #animal_densities_temp = np.random.rand(self.gridsize,self.gridsize)
        river_dist = [72 , 51 , 22 , 48 , 0 , 64 , 56 , 70 , 1 , 30 , 44 , 81 , 79 , 23 , 66 , 19 , 33 , 50 , 13 , 39 , 15 , 19 , 79 , 91 , 16 , 93 , 60 , 42 , 46 , 26 , 11 , 1 , 30 , 83 , 15 , 98 , 74 , 46 , 57 , 38 , 76 , 28 , 70 , 31 , 93 , 67 , 37 , 92 , 72 , 30 , 9 , 11 , 20 , 44 , 38 , 20 , 5 , 90 , 49 , 57 , 19 , 0 , 33 , 50 , 56 , 98 , 43 , 21 , 5 , 3 , 2 , 17 , 76 , 90 , 65 , 6 , 86 , 82 , 56 , 90 , 73 , 95 , 61 , 66 , 83 , 2 , 42 , 93 , 24 , 41 , 39 , 52 , 97 , 39 , 91 , 37 , 38 , 53 , 71 , 68 , 56 , 0 , 45 , 79 , 48 , 74 , 37 , 71 , 56 , 38 , 68 , 6 , 6 , 96 , 79 , 69 , 41 , 54 , 76 , 13 , 4 , 36 , 48 , 31 , 37 , 47 , 2 , 39 , 76 , 17 , 29 , 3 , 34 , 42 , 60 , 11 , 0 , 29 , 96 , 55 , 64 , 34 , 74 , 39 , 20 , 69 , 1 , 50 , 52 , 53 , 85 , 36 , 10 , 93 , 55 , 99 , 56 , 6 , 32 , 84 , 91 , 14 , 11 , 88 , 37 , 67 , 18 , 61 , 22 , 48 , 55 , 59 , 14 , 32 , 73 , 16 , 17 , 69 , 13 , 88 , 85 , 78 , 80 , 44 , 39 , 26 , 76 , 59 , 82 , 73 , 94 , 31 , 82 , 82 , 54 , 45 , 5 , 77 , 3 , 53 , 72 , 70 , 40 , 37 , 4 , 75 , 31 , 70 , 21 , 13 , 0 , 50 , 77 , 95 , 20 , 90 , 73 , 96 , 0 , 8 , 89 , 11 , 64 , 65 , 5 ]
        river_dist = np.array(river_dist)
        boundary_dist = np.array([82 , 2 , 40 , 68 , 5 , 38 , 55 , 21 , 4 , 49 , 83 , 0 , 30 , 88 , 20 , 5 , 93 , 36 , 38 , 41 , 41 , 90 , 73 , 12 , 35 , 72 , 18 , 20 , 25 , 72 , 75 , 14 , 8 , 66 , 41 , 83 , 78 , 70 , 8 , 55 , 39 , 59 , 56 , 26 , 74 , 4 , 80 , 56 , 84 , 44 , 93 , 73 , 46 , 49 , 61 , 33 , 67 , 70 , 82 , 45 , 79 , 9 , 35 , 79 , 81 , 21 , 89 , 83 , 74 , 87 , 55 , 76 , 12 , 30 , 33 , 27 , 91 , 18 , 85 , 93 , 10 , 50 , 43 , 98 , 3 , 7 , 13 , 32 , 47 , 0 , 77 , 74 , 51 , 43 , 95 , 89 , 67 , 92 , 23 , 25 , 87 , 24 , 69 , 32 , 24 , 69 , 83 , 15 , 61 , 89 , 36 , 68 , 31 , 18 , 34 , 39 , 50 , 75 , 95 , 78 , 28 , 28 , 19 , 11 , 64 , 54 , 91 , 8 , 26 , 58 , 74 , 23 , 43 , 65 , 82 , 4 , 74 , 13 , 84 , 50 , 69 , 81 , 98 , 53 , 64 , 73 , 30 , 94 , 97 , 28 , 82 , 36 , 95 , 4 , 39 , 28 , 75 , 51 , 31 , 37 , 72 , 13 , 4 , 44 , 9 , 49 , 77 , 75 , 16 , 87 , 14 , 56 , 16 , 65 , 80 , 7 , 31 , 37 , 71 , 8 , 0 , 51 , 53 , 2 , 7 , 84 , 19 , 22 , 32 , 29 , 15 , 39 , 10 , 23 , 88 , 11 , 36 , 60 , 88 , 5 , 55 , 32 , 86 , 32 , 41 , 98 , 12 , 65 , 27 , 3 , 33 , 73 , 80 , 49 , 31 , 74 , 62 , 85 , 70 , 56 , 41 , 58 , 44 , 77 , 29 ])
        road_dist = np.array([86 , 39 , 52 , 62 , 28 , 39 , 26 , 87 , 74 , 96 , 68 , 28 , 39 , 41 , 52 , 93 , 50 , 26 , 15 , 30 , 27 , 74 , 79 , 93 , 60 , 74 , 84 , 89 , 49 , 43 , 20 , 37 , 52 , 86 , 59 , 92 , 99 , 73 , 97 , 12 , 10 , 60 , 68 , 80 , 38 , 25 , 67 , 53 , 14 , 59 , 94 , 38 , 56 , 20 , 16 , 83 , 41 , 43 , 48 , 17 , 17 , 54 , 51 , 56 , 29 , 42 , 72 , 53 , 85 , 21 , 68 , 67 , 40 , 83 , 73 , 37 , 61 , 8 , 23 , 53 , 6 , 47 , 9 , 88 , 92 , 9 , 64 , 73 , 32 , 3 , 30 , 22 , 31 , 19 , 22 , 35 , 65 , 82 , 18 , 16 , 39 , 23 , 78 , 44 , 67 , 61 , 18 , 34 , 79 , 59 , 29 , 89 , 94 , 6 , 87 , 76 , 70 , 52 , 62 , 0 , 28 , 94 , 64 , 96 , 82 , 28 , 90 , 90 , 99 , 4 , 57 , 6 , 0 , 84 , 92 , 51 , 89 , 87 , 94 , 92 , 75 , 72 , 99 , 18 , 8 , 83 , 6 , 26 , 74 , 60 , 69 , 59 , 58 , 83 , 1 , 51 , 95 , 69 , 81 , 73 , 64 , 67 , 34 , 0 , 62 , 41 , 32 , 4 , 84 , 57 , 14 , 80 , 25 , 51 , 15 , 5 , 85 , 9 , 33 , 3 , 47 , 47 , 96 , 96 , 76 , 53 , 75 , 1 , 24 , 4 , 4 , 78 , 94 , 72 , 80 , 20 , 36 , 77 , 12 , 47 , 25 , 91 , 23 , 80 , 37 , 99 , 48 , 46 , 27 , 4 , 34 , 93 , 2 , 90 , 90 , 73 , 12 , 66 , 21 , 11 , 13 , 18 , 22 , 97 , 98 ])
        order_r = river_dist.argsort()
        ranks_r = order_r.argsort()

        order_b = boundary_dist.argsort()
        ranks_b = order_b.argsort()

        order_rd = road_dist.argsort()
        ranks_rd = order_rd.argsort()

        awr = np.zeros(225)
        for i in range (225):
          awr[i] = 0.8*ranks_r[i] + 0.1*ranks_b[i] + 0.1*ranks_rd[i]

        order_afr = awr.argsort()
        afr = order_afr.argsort()

        pwr = np.zeros(225)
        for i in range (225):
          pwr[i] = 0.7*afr[i] + 0.05*ranks_r[i] + 0.15*ranks_b[i] + 0.1*ranks_rd[i]

        order_pfr = pwr.argsort()
        
        pfr = order_pfr.argsort()
        
        afr = 225 - afr
        pfr = 225 - pfr


        
        afr = np.reshape(afr,(15,15))
        pfr = np.reshape(pfr,(15,15))

        self.afr_temp = afr/225.0



        self.pfr_temp = pfr/225.0


        #print(self.pfr_temp[:,:])
        self.animal_densities = self.pfr_temp.copy()  
        self.animal_densities_initial = self.pfr_temp.copy()  #CONSTANT THROUGHOUT ALL EPISODES
        self.poacher_reward = 0
        self.reached_int_target = False

 
        self.visits = np.zeros((self.gridsize,self.gridsize))
        
        
        
        
    # return all entities in the world
    @property
    def entities(self):
        return self.agents + self.landmarks

    # return all agents controllable by external policies
    @property
    def policy_agents(self):  #SURVEILLANCE DRONE
        return [agent for agent in self.agents if agent.action_callback is None]

    # return all agents controlled by world scripts
    @property
    def scripted_agents(self):   #ATTACKER AND  RANGER
        return [agent for agent in self.agents if agent.action_callback is not None]

    # update state of the world
    def step(self):
        # set actions for scripted agents 
        #for agent in self.scripted_agents:
        #    self.scripted_action(agent)
            

        self.time = self.time + 1  #UPDATE TIME STEP   
     
        for agent in self.policy_agents:
            self.update_agent_state(agent)
        for agent in self.scripted_agents:
            self.scripted_action(agent)    
        for agent in self.scripted_agents:

            self.update_agent_state(agent)


  


    def inside_grid(self,pos):
      
      if(pos[0] < 0 or pos[0] > (self.gridsize-1) or pos[1] < 0 or pos[1] > (self.gridsize-1)):
        return False
      else:
        return True 

    '''def move_to_target_pos(self,agent,target): 

       
            #DETERMINES WHICH ACITON TO TAKE TO REACH THE TARGET COVERING SHORTEST PATH GREEDILY
      a = abs((agent.state.p_pos[0]-1) - target[0]) + abs(agent.state.p_pos[1] - target[1])
      b = abs((agent.state.p_pos[0]+1) - target[0]) + abs(agent.state.p_pos[1] - target[1])
      c = abs(agent.state.p_pos[0] - target[0]) + abs((agent.state.p_pos[1]-1) - target[1])
      d = abs(agent.state.p_pos[0] - target[0]) + abs((agent.state.p_pos[1]+1) - target[1])
      minimum = min(a,b,c,d)
      print(minimum)
      
      if(minimum == a):
        corrected = agent.state.p_pos + np.array([-1,0])
        
        if(self.inside_grid(corrected)):
           agent.action.u[0] =-1
           agent.action.u[1] = 0
           return
      if(minimum == b):
        corrected = agent.state.p_pos + np.array([1,0])
        
        if(self.inside_grid(corrected)):
           agent.action.u[0] = 1
           agent.action.u[1] = 0
           return
      if(minimum == c):
        corrected = agent.state.p_pos + np.array([0,-1])
        
        if(self.inside_grid(corrected)):
           agent.action.u[1] =-1
           agent.action.u[0] = 0
           return
      if(minimum == d):
        corrected = agent.state.p_pos + np.array([0,1])
        
        if(self.inside_grid(corrected)):
           agent.action.u[1] = 1 
           agent.action.u[0] = 0
           return'''
      

   

    def flee(self,agent):
      
      pos = agent.state.p_pos
      a = abs(pos[0]-0)
      b = abs(pos[0]-self.gridsize-1)
      c = abs(pos[1]-0)
      d = abs(pos[1]-self.gridsize-1)

      mm = min(a,b,c,d)
      
      if(mm == a):
        agent.action.u[0] = -1
        agent.action.u[1] = 0
        return
      if(mm == b):
        agent.action.u[0] =  1
        agent.action.u[1] = 0
        return
      if(mm == c):
        agent.action.u[1] = -1
        agent.action.u[0] = 0
        return
      if(mm == d):
        agent.action.u[1] =  1
        agent.action.u[0] = 0
        return    



    def reached_poacher_target_reward(self,agent):
      
      if(self.poacher_reward >= 200):
        self.reached_target = True

    def reached_poacher_int_target(self,agent):
      
      if(agent.state.p_pos[0] == self.target[0] and agent.state.p_pos[1] == self.target[1]):
        self.reached_int_target = True
        

    '''def reached_poacher_target(self,agent):
      
      if(agent.state.p_pos[0] == self.target[0] and agent.state.p_pos[1] == self.target[1]):
        self.reached_target = True'''
      

    


      
    def scripted_action(self, agent):
      self.reached_poacher_int_target(agent)
    

      

      if(self.reached_int_target and self.target is not None): # UPDATE FOR INTERMEDIATE TARGETS


        self.targets.append(self.target.copy())
    
        self.reached_int_target = False


        #####################  ADDING ANIMAL DENSITIES #####################

        ranks_temp = rankdata(self.poacher_policy.copy().reshape((self.gridsize*self.gridsize)),method = 'ordinal').reshape((self.gridsize,self.gridsize))

        for i in range(self.gridsize):
          for j in range(self.gridsize):
            if(abs(self.poachers[0].state.p_pos[0] - i) > 1 or abs(self.poachers[0].state.p_pos[1] - j) > 1):
              ranks_temp[i][j] = 0

        p, q = np.where( ranks_temp == np.amax(ranks_temp))        
        self.target[0] = p
        self.target[1] = q



          
        ##################################################################
        self.poacher_reward = self.poacher_reward + 1
        self.poacher_policy[int(self.target[0])][int(self.target[1])] = 0.0  
        
        
      self.reached_poacher_target_reward(agent)
      #self.saw_drone_target(self.agents[0])
      #self.reached_poacher_target(self.agents[2])
 

      if(agent.name == "poacher"):

        if(self.premature_flee or self.reached_target):
          if(self.time == (self.flee_time + 1)):
       
            self.flee_pos = agent.state.p_pos
           
          self.flee(agent)
          
        else:

          agent.action.u[0] = self.target[0]
          agent.action.u[1] = self.target[1]
         
          #self.move_to_target_pos(agent,self.target)
 
            
        



    def update_agent_state(self,agent): 
         
        if(agent.name == "poacher" and not self.premature_flee and not self.reached_target):
          agent.state.p_pos = agent.action.u        



        else:  

        
          agent.state.p_pos = agent.state.p_pos +  agent.action.u
          if(agent.state.p_pos[0] > (self.gridsize-1)):
            agent.state.p_pos[0] = self.gridsize-1
          if( agent.state.p_pos[1] > (self.gridsize-1)):
            agent.state.p_pos[1] = self.gridsize-1
          if(agent.state.p_pos[0] < 0):
            agent.state.p_pos[0] = 0
          if(agent.state.p_pos[1] < 0):
            agent.state.p_pos[1] = 0  

          agent.state.signal = agent.action.signal

          agent.state.warn = agent.action.warn  

        if(agent.state.p_pos[0] < 0 or agent.state.p_pos[1] < 0):
          print("ANOMALY!")


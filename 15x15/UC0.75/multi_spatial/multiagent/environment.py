



import gym
from gym import spaces
from gym.envs.registration import EnvSpec
import numpy as np
from multiagent.multi_discrete import MultiDiscrete
# environment for all agents in the multiagent world/
# currently code assumes that no agents will be created/destroyed at runtime!

#For reference, instantiated as: env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)    World and scenario are created in advance. Refer to make_env.py
class MultiAgentEnv(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array']
    }

    def __init__(self, world, reset_callback=None, reward_callback=None, shared_reward_callback = None,
                 observation_callback=None, info_callback=None,
                 done_callback=None, shared_viewer=True):   #shared_viewer seems to be related to rendering

        self.world = world
        self.agents = self.world.policy_agents
        # set required vectorized gym env property
        self.n = len(world.policy_agents)
        # scenario callbacks
        self.reset_callback = reset_callback  #for resetting world
        self.reward_callback = reward_callback  #for individual reward function
        self.shared_reward_callback = shared_reward_callback #for shared reward for defender agents
        self.observation_callback = observation_callback  #for drone observation
        self.info_callback = info_callback   #for benchmarking data
        self.done_callback = done_callback    #I don't think we use this
        # environment parameters
        self.discrete_action_space = True
        # if true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector
        self.discrete_action_input = True   


        


        

        
        self.action_space = []
        self.observation_space = []
        for agent in self.agents:  
            
            
            if self.discrete_action_space and agent.name == "drone":  
                u_action_space = spaces.Discrete(15)
            elif self.discrete_action_space and agent.name == "ranger":
                u_action_space = spaces.Discrete(5)    
            else:
                u_action_space = spaces.Box(low=-agent.u_range, high=+agent.u_range, shape=(world.dim_p,), dtype=np.float32)
            
            if agent.movable:
                
                self.action_space.append(u_action_space)
            
            
            # communication action space   NOTE : WE DON'T NEED THIS FOR NOW
            '''if self.discrete_action_space:
                c_action_space = spaces.Discrete(world.dim_c)
            else:
                c_action_space = spaces.Box(low=0.0, high=1.0, shape=(world.dim_c,), dtype=np.float32)
            if not agent.silent:
                total_action_space.append(c_action_space)'''


            

            
          


            self.observation_space.append(np.array([spaces.Discrete(2),spaces.Discrete(2)])) 
         
            agent.action.c = np.zeros(self.world.dim_c) #RELATED TO COMMUNICATION. IGNORE

        # rendering
        self.shared_viewer = shared_viewer
        if self.shared_viewer:
            self.viewers = [None]
        else:
            self.viewers = [None] * self.n
        self._reset_render()

    def step(self, action_n):   #action_n is probably a list containing actions of each agent
        

        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}
        self.agents = self.world.policy_agents  #IN OUR CASE, THE DRONE AND THE RANGER
        # set action for each agent
        
        for i, agent in enumerate(self.agents):
            
            self._set_action(action_n[i], agent, self.action_space[i])    
               

       
        self.world.step()   



        for agent in self.agents:
            obs_n.append(self._get_obs(agent))  #S_t+1 
              
            reward_n.append(self._get_reward(agent))  #agent specific rewards  
                 
            info_n['n'].append(self._get_info(agent))

        final_rewards = []

        
        shared_reward = self._get_shared_rew()
        done_n.append(self._get_done(agent))
        for item in reward_n:
            
            final_rewards.append(item + shared_reward)
                
        
        for agent in self.world.agents:
          agent.action.u = np.zeros(2)
          agent.action.warn = False
          
          agent.action.signal = False

        for agent in self.world.policy_agents:
            self.world.visits[int(agent.state.p_pos[0])][int(agent.state.p_pos[1])] = 1
            self.world.v_counts[int(agent.state.p_pos[0])][int(agent.state.p_pos[1])] = self.world.v_counts[int(agent.state.p_pos[0])][int(agent.state.p_pos[1])] + 1
        
        for agent in self.world.poachers:
            if(agent.state.p_pos[0] != -99):
                self.world.poacher_counts[int(agent.state.p_pos[0])][int(agent.state.p_pos[1])] = self.world.poacher_counts[int(agent.state.p_pos[0])][int(agent.state.p_pos[1])] + 1 + self.world.time*0.02
        
        return obs_n, final_rewards, done_n, info_n


    
    
    
    
    def reset(self, d, r, p):
        # reset world
        self.reset_callback(self.world, d, r, p)
        # reset renderer
        self._reset_render()
        # record observations for each agent
        obs_n = []
        self.agents = self.world.policy_agents
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
        return obs_n

    
    # get info used for benchmarking
    def _get_info(self, agent):
        if self.info_callback is None:
            return {}
        return self.info_callback(agent, self.world)

    
    # get observation for a particular agent
    def _get_obs(self, agent):
        if self.observation_callback is None:
            return np.zeros(0)
        return self.observation_callback(agent, self.world)

 
    def _get_done(self, agent):                           
        if self.done_callback is None:
            return False
        return self.done_callback(self.world)

    # get reward for a particular agent
    def _get_reward(self, agent):
        if self.reward_callback is None:
            return 0.0
        return self.reward_callback(agent, self.world)

    def _get_shared_rew(self):
        return self.shared_reward_callback(self.world)    

    # set env action for a particular agent
    def _set_action(self, action, agent, action_space, time=None):
       agent.action.u = np.zeros(self.world.dim_p)
       agent.action.c = np.zeros(self.world.dim_c)
                
       if(agent.name == "drone"):

        if agent.movable:

            if self.discrete_action_input:
                
                
                agent.action.u = np.zeros(self.world.dim_p)

                # process discrete action
                if action == 0: 
                    agent.action.u[1] = +1.0  #UP,NOOP   
                if action == 1: 
                  agent.action.u[1] = +1.0  #UP,WARN
                  agent.action.warn = True
                if action == 2: 
                  agent.action.u[1] = +1.0  #UP,SIGNAL
                  agent.action.signal = True


                if action == 3: agent.action.u[1] = -1.0  #DOWN,NOOP
                if action == 4: 
                  agent.action.u[1] = -1.0  #DOWN,WARN
                  agent.action.warn = True
                if action == 5: 
                  agent.action.u[1] = -1.0  #DOWN,SIGNAL
                  agent.action.signal = True

                if action == 6: agent.action.u[0] = -1.0  #LEFT NOOP
                if action ==7: 
                  agent.action.u[0] = -1.0  #LEFT WARN
                  agent.action.warn = True
                if action == 8: 
                  agent.action.u[0] = -1.0 #LEFT SIGNAL
                  agent.action.signal = True


                if action == 9: agent.action.u[0] = +1.0 #RIGHT NOOP
                if action == 10: 
                  agent.action.u[0] = +1.0 #RIGHT WARN
                  agent.action.warn = True
                if action == 11:
                   agent.action.u[0] = +1.0 #RIGHT SIGNAL
                   agent.action.signal = True

                if action == 12: agent.action.u[0] =  0.0 #STAY
                if action == 13: 
                  agent.action.u[1] =  0.0 #STAY WARN
                  agent.action.warn = True
                if action == 14: 
                  agent.action.u[1] =  0.0 #STAY SIGNAL
                  agent.action.signal = True


       if(agent.name == "ranger"):

        if agent.movable:

            if self.discrete_action_input:
                
                
                agent.action.u = np.zeros(self.world.dim_p)
                
                # process discrete action
                if action == 0: agent.action.u[1] = +1.0  #UP,NOOP  
                if action == 1: agent.action.u[1] = -1.0  #DOWN,NOOP
                if action == 2: agent.action.u[0] = -1.0  #LEFT NOOP
                if action == 3: agent.action.u[0] = +1.0 #RIGHT NOOP
                if action == 4: agent.action.u[0] =  0.0 #STAY










        '''if not agent.silent:       #FOR US,AGENT.SILENT IS FALSE
            # communication action
            if self.discrete_action_input:
                agent.action.c = np.zeros(self.world.dim_c)
                agent.action.c[action[0]] = 1.0
            else:
                agent.action.c = action[0]
            action = action[1:]'''



    





    ################################################### RENDERING FUNCTIONS ##############################################################
    # reset rendering assets
    def _reset_render(self):
        self.render_geoms = None
        self.render_geoms_xform = None

    # render environment
    def render(self, mode='human'):        
        if mode == 'human':
            alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            message = ''
            for agent in self.world.agents:
                comm = []
                for other in self.world.agents:
                    if other is agent: continue
                    if np.all(other.state.c == 0 or other.state.c ==  None):
                        word = '_'
                    else:
                        word = alphabet[np.argmax(other.state.c)]
                    message += (other.name + ' to ' + agent.name + ': ' + word + '   ')
            print(message)

        for i in range(len(self.viewers)):
            # create viewers (if necessary)
            if self.viewers[i] is None:
                # import rendering only if we need it (and don't import for headless machines)
                #from gym.envs.classic_control import rendering
                from multiagent import rendering
                self.viewers[i] = rendering.Viewer(300,300)   # 700 x 700 is the width and height of display. See rendering.py in the repo

        # create rendering geometry
        
        if self.render_geoms is None:
            
            # import rendering only if we need it (and don't import for headless machines)    <------------- THIS ONE SEEMS TO BE FOR RENDERING THE ENTITIES 
            #from gym.envs.classic_control import rendering
            from multiagent import rendering
            self.render_geoms = []
            self.render_geoms_xform = []
            for entity in self.world.entities:
                geom = rendering.make_circle(entity.size)
                xform = rendering.Transform()
                
                if 'drone' in entity.name:
                    
                   
                    if(entity.state.signal ==  True):
                        
                       
                        geom.set_color(*entity.signal_color, alpha=0.5)
                    elif(entity.state.warn ==  True):
                     
                        
                        geom.set_color(*entity.warn_color, alpha=0.5)
                    else:  
                      
                        geom.set_color(*entity.color, alpha=0.5)
                else:
                 
                    geom.set_color(*entity.color)
                geom.add_attr(xform)
                self.render_geoms.append(geom)
                self.render_geoms_xform.append(xform)

            # add geoms to viewer
            for viewer in self.viewers:
                viewer.geoms = []
                for geom in self.render_geoms:
                    viewer.add_geom(geom)

        results = []
        for i in range(len(self.viewers)):
            from multiagent import rendering
            # update bounds to center around agent
            cam_range = 8
            if self.shared_viewer:
                pos = np.array([7,7])#self.world.agents[0].state.p_pos #np.zeros(self.world.dim_p)
            else:
                pos = self.agents[i].state.p_pos
            self.viewers[i].set_bounds(pos[0]-cam_range,pos[0]+cam_range,pos[1]-cam_range,pos[1]+cam_range)
            # update geometry positions
            for e, entity in enumerate(self.world.entities):
                
                self.render_geoms_xform[e].set_translation(*entity.state.p_pos)
            # render to display or array
            results.append(self.viewers[i].render(return_rgb_array = mode=='rgb_array'))

        return results

    # create receptor field locations in local coordinate frame
    def _make_receptor_locations(self, agent):
        receptor_type = 'grid'  #'polar'
        range_min = 0.05 * 2.0
        range_max = 1.00
        dx = []
        # circular receptive field
        if receptor_type == 'polar':
            for angle in np.linspace(-np.pi, +np.pi, 8, endpoint=False):
                for distance in np.linspace(range_min, range_max, 3):
                    dx.append(distance * np.array([np.cos(angle), np.sin(angle)]))
            # add origin
            dx.append(np.array([0.0, 0.0]))
        # grid receptive field
        if receptor_type == 'grid':
            for x in np.linspace(-range_max, +range_max, 5):
                for y in np.linspace(-range_max, +range_max, 5):
                    dx.append(np.array([x,y]))
        return dx

# vectorized wrapper for a batch of multi-agent environments
# assumes all environments have the same observation and action space
'''class BatchMultiAgentEnv(gym.Env):            #<------------------------ DON'T THINK THIS IS NEEDED RIGHT NOW
    metadata = {
        'runtime.vectorized': True,
        'render.modes' : ['human', 'rgb_array']
    }

    def __init__(self, env_batch):
        self.env_batch = env_batch

    @property
    def n(self):
        return np.sum([env.n for env in self.env_batch])

    @property
    def action_space(self):
        return self.env_batch[0].action_space

    @property
    def observation_space(self):
        return self.env_batch[0].observation_space

    def step(self, action_n, time):
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}
        i = 0
        for env in self.env_batch:
            obs, reward, done, _ = env.step(action_n[i:(i+env.n)], time)
            i += env.n
            obs_n += obs
            # reward = [r / len(self.env_batch) for r in reward]
            reward_n += reward
            done_n += done
        return obs_n, reward_n, done_n, info_n

    def reset(self):
        obs_n = []
        for env in self.env_batch:
            obs_n += env.reset()
        return obs_n

    # render environment
    def render(self, mode='human', close=True):
        results_n = []
        for env in self.env_batch:
            results_n += env.render(mode, close)
        return results_n'''


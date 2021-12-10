#This module defines a maze with two terminal states.

import numpy as np
import utils


class Maze:
    def __init__(self):
        """
        Initalize the maze object
        """
        reward = 0
        state = 0
        termination = 0
        
        self.observation = (reward, state, termination)
    
    def env_init(self, env_info={}):
        """
        Initializes the maze according to the parameters in env_info
        """
        
        #Set up the maze
        self.width = env_info['width']
        self.height = env_info['height']
        self.walls = [utils.matrix_to_ndarray(wall, self.width) for wall in env_info['walls']]
        
        #Get infos
        self.num_states = env_info['width']*env_info['height']
        
        
        #Initalize the task – flatten the matrix
        self.start_state = utils.matrix_to_ndarray(env_info['start_state'], self.width)
        self.reward_states = [utils.matrix_to_ndarray(state, self.width) for state in env_info['reward_states']]
        self.rewards = env_info['rewards']
        
    
    def env_start(self):
        """
        Get first observation
        """
        reward = 0
        state = self.start_state
        termination = False
        
        self.observation = (reward, state, termination)
        
        return self.observation
    
    def env_step(self, action):
        """
        Take action and observe the reward signal
        """
        reward = 0
        last_state = self.observation[1]
        termination = False
        
        #Observe new state
        if action == 0:
            #Go up
            state = last_state + self.width
            
            
        elif action == 1:
            #Go right
            state = last_state + 1
            
        elif action == 2:
            #Go down
            state = last_state - self.width
            
        elif action == 3:
            #Go left
            state = last_state - 1
            
        
        #Compare if the new state is valid, if not reset
        if state in self.walls:
            state = last_state
            
        elif state not in range(self.num_states):
            raise ValueError(f"State {state} is not on the grid!")
        
            
        
        #Check if the state is a terminal state
        if state in self.reward_states:
            state_index = self.reward_states.index(state)
            
            reward = self.rewards[state_index]
            termination = True
            
        self.observation = (reward, state, termination)
        
        
        return self.observation
    
    def env_reset(self):
        """
        Wrapper function to avoid confusion
        """
        return self.env_start()
    
    def env_plot(self):
        """
        Plot the current maze
        """
        
        walls = [utils.ndarray_to_matrix(wall, self.width) for wall in self.walls]
        
        start_state = utils.ndarray_to_matrix(self.start_state, self.width)
        
        reward_states = [utils.ndarray_to_matrix(state, self.width) for state in self.reward_states]
        
        current_state = utils.ndarray_to_matrix(self.observation[1], self.width)
        
        utils.plot((self.width, self.height), walls, start_state, reward_states, current_state)
        
        
    def env_change_task(self, start_state=None, reward_states=None, rewards=None):
        """
        Change the task by moving the start_state, reward_state or changing the rewards.
        """
        
        if start_state:
            self.start_state = utils.matrix_to_ndarray(start_state, self.width)
            
        if reward_states:
            self.reward_states = [utils.matrix_to_ndarray(state, self.width) for state in reward_states]
            
        if rewards:
            self.rewards = rewards
            
        pass
        
        
    def env_change_env(self, walls):
        """
        Add or remove walls to change the environment. 
        """
        self.walls = [utils.matrix_to_ndarray(wall, self.width) for wall in walls]
        pass
        
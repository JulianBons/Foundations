#This module defines agents that can be used for testing in the notebook

import numpy as np
import utils

class QAgent:
    
    def __init__(self):
        """
        Creates an instance of an agent using the Q update to compute the value functions
        """
        pass
    
    def agent_init(self, agent_info={}):
        """
        Initalizes the agents and sets up the task.
        """
        #Set up tracking variables
        self.last_action = None
        self.num_steps = 0
        self.num_episodes = 0
        self.last_state = None

        
        #Set up task
        self.num_action = agent_info['num_actions']
        
        #Set up hyperparameter
        self.step_size = agent_info['step_size']
        self.discount_factor = agent_info['discount_factor']
        self.epsilon = agent_info['epsilon']
    
        self.q_values = np.zeros((agent_info['num_states'], agent_info['num_actions']))
        
        pass
    
    def agent_start(self, start_state):
        """
        Sample the first action
        """
        if np.random.random()<self.epsilon:
            action = np.random.choice(range(4))
        else:
            action = utils.argmax(self.q_values[start_state])
            
        
        #Safe the last state and last action
        self.last_action = action
        self.last_state = start_state
        
        self.num_steps = 1
            
        return action
            
            
    
    def agent_step(self, reward, state):
        """
        Update the Q-values and choose an action for the given state 
        """
        
        #Update the q-value
        self.q_values[self.last_state, self.last_action] += self.step_size*(
            reward + self.discount_factor*np.max(self.q_values[state]) - 
                       self.q_values[self.last_state, self.last_action])
        
        #Sample an action using epsilon-greedy Q-Learning
        if np.random.random()<self.epsilon:
            action = np.random.choice(range(4))
        else:
            action = utils.argmax(self.q_values[state])
            
                       
        #Safe the last state and action
        self.last_action = action
        self.last_state = state
        
        self.num_steps += 1
            
        return action 
        
        
    
    def agent_stop(self, reward, state):
        """
        Update the Q-Value and stop the agent
        """
        
        #Update the q-value
        self.q_values[self.last_state, self.last_action] += self.step_size*(
            reward + self.discount_factor*np.max(self.q_values[state]) - 
                       self.q_values[self.last_state, self.last_action])
        
        
        #Safe the last state
        self.last_state = state
        
        self.num_steps += 1
        self.num_episodes += 1
        
        pass
    
    def agent_reset(self, state):
        """
        Wrapper function to avoid confusion
        """
        return self.agent_start(self, state)
    
    
    
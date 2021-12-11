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

    
    
    
    
class TemporalDifferenceSuccessor:
    def __init__(self):
        """
        Creates an instances of an agent that uses SR.
        """
        pass
    
    def agent_init(self, agent_info={}):
        """
        Initlaizes the SR agent according to the parameters defined in agent_info
        """
        #Set up tracking variables
        self.last_action = None
        self.last_state = None
        self.num_steps = 0
        
        #Set up task
        self.num_states = agent_info['num_states']
        self.num_actions = agent_info['num_actions']
        
        #Set up matrices
        self.Q = np.zeros([self.num_states, self.num_actions])
        self.M = np.zeros([self.num_states, self.num_states])
        self.w = np.zeros([self.num_states]) #Reward_vector
        
        
        #Set Hyperparameter
        self.step_size = agent_info['step_size']
        self.discount_factor = agent_info['discount_factor']
        self.epsilon = agent_info['epsilon']
        
        
    def agent_sample_action(self, state):
        """
        Sample action for the given state
        """
        if np.random.random()<self.epsilon: #Exploration
            action = np.random.choice(range(4))
            
        else: #Exploitation
            action = utils.argmax(self.Q[state])
            
        return action
    
    
    def agent_update_values(self, reward, state):
        """
        Updates the Q-values, reward_vector and the SR
        """
        
        #Compute w
        delta = reward + self.discount_factor*self.w[state] - self.w[self.last_state]
        self.w[self.last_state] += self.step_size*delta
        
        #Compute the SR-error and update M
        one_hot = np.zeros(self.num_states)
        one_hot[self.last_state] = 1

        delta = one_hot + self.discount_factor*self.M[state] - self.M[self.last_state]
        self.M[self.last_state] += self.step_size*delta


        #Update Q-values
        self.Q[self.last_state][self.last_action] = np.dot(self.M[state], self.w)

        pass
        
        
    
    def agent_start(self, start_state):
        """
        Samples first action for the starting_state
        """
        
        action = self.agent_sample_action(start_state)
        
        #Store last state and action
        self.last_state = start_state
        self.last_action = action
        
        
        #Update tracking variables
        self.num_steps += 1
        
        return action
        
    
    def agent_step(self, reward, state):
        """
        Sample action in the given state. 
        Update the SR in the agent.
        """ 
        
        #Update the values
        self.agent_update_values(reward, state)   
            
        #Sample action
        action = self.agent_sample_action(state)  
        
        
        #Store last state and action
        self.last_state = state
        self.last_action = action
        
        
        #Update tracking variables
        self.num_steps += 1
        
        return action
        
    
    def agent_stop(self, reward, state):
        """
        Updates the agent after reaching the final state
        """
        
        self.agent_update_values(reward, state)
        
        #Store last_state
        self.last_state = state
        
        #Update tracking variables
        self.num_steps += 1
        
        pass
        
        
        
    
    
    
#This module provides wrapper functions for the RL task. 

import numpy as np
import utils
from env import Maze
from agent import QAgent

class ReinforcementLearning:
    def __init__(self, env, agent):
        """
        Initilizes the agent and the environment
        Params:
            env -- env object as defined in env.py
            agent -- agent object as defined in agent.py 
                Can be QAgent, SRAgent or LRLAgent
        """
        self.env = env()
        self.agent = agent()
        
        pass
        
    
    def rl_init(self, env_info={}, agent_info={}):
        """
        Initlizies the environment and agent according to the given settings
        Params:
            env_info -- dict, 
                width -- int, width of the environment
                height -- int, height of the environment
                walls -- list with the walls of the environment
                start_state -- tuple, row and col of the start state
                reward_states -- list of tuples, row and col of the reward states
                rewards -- list of ints, reward for each reward state
                
            agent_info -- dict,
                num_states -- int, number of total states (width*height)
                num_actions -- int, number of actions
                step_size -- float, alpha in Q-update 
                discount-factor -- float in range 0 to 1, gamma in Q-Update
                epsilon -- float in range 0 to 1, percentage of exploration
        """
        #Initalize env and agent
        self.env.env_init(env_info)
        self.agent.agent_init(agent_info)
        
        #Set up tracking variables over episode
        self.total_reward = 0
        self.num_episodes = 0
        
        pass
        
    
    def rl_start(self):
        """
        Starts the interaction. The first state and environment are sampled.
        """
        #Set up tracking variables for each episode
        self.num_steps = 0
        self.trajectory = []
        
        #Start the interaction
        (reward, state, termination) = self.env.env_start()
        action = self.agent.agent_start(state)
        
        #Track the changes
        self.num_episodes += 1
        self.num_steps += 1
        self.trajectory.append(state)
        
        return (reward, state, action, termination)

    
    def rl_step(self, reward, state, action):
        """
        Observe the effect of the interaction.
        """
        #Interaction
        (reward, state, termination) = self.env.env_step(action)
        action = self.agent.agent_step(reward, state)
        
        #Track the changes
        self.num_steps += 1
        self.trajectory.append(state)
        
        return (reward, state, action, termination)
        
        
    
    def rl_stop(self, reward, state):
        """
        Final update of the agent
        """
        self.agent.agent_stop(reward, state)
        
        #Track the changes
        self.num_steps += 1
        self.trajectory.append(state)
        self.total_reward += reward
        
        pass
        
    
    def rl_episode(self):
        """
        Simulate a full episode
        """
        termination = False
        
        (reward, state, action, termination) = self.rl_start()
        
        
        while not termination:
            (reward, state, action, termination) = self.rl_step(reward, state, action)
            
        self.rl_stop(reward, state)
        
        return termination
                
    
    def rl_change_task(self, start_state=None, reward_states=None, rewards=None):
        """
        Change the task by moving the start_state, reward_state or changing the rewards.
        """
        self.env.env_change_task(start_state, reward_states, rewards)
            
        pass
        
        
    
    def rl_change_env(self, walls):
        """
        Add or remove walls to change the environment. 
        Walls contains a list containing the locations of the walls as tuples. 
        """
        self.env.env_change_env(walls)
        
        pass
    
    
    def rl_plot(self):
        """
        Plot the current_environment
        """
        self.env.env_plot()
        pass

    
    
    
    
    



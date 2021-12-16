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

    
    def rl_step(self, reward, last_state, last_action):
        """
        Observe the effect of the interaction.
        """
        #Interaction
        (reward, state, termination) = self.env.env_step(last_action)
        
        if self.agent.replay:
            self.agent.memory[(last_state, last_action)] = (reward, state)

        
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
    
    
    def rl_change_epsilon(self, epsilon):
        """
        Changes the epsilon parameter of the agent. 
        If set to 1, the agent will employ a random policy. 
        """
        self.agent.epsilon = epsilon
        pass
    
    def rl_change_values(self, state, value):
        """
        To simulate revaluations. 
        """
        self.agent.w[state] = value
        pass
    
    
    def rl_explore_env(self, num_steps):
        """
        Explore the maze with no task and no rewards
        """
        #Store old values
        old_epsilon = self.agent.epsilon
        old_reward_states = [utils.ndarray_to_matrix(state, self.env.width) for state in self.env.reward_states]
        old_rewards = self.env.rewards
        old_replay = self.agent.replay
     
        #Use a random policy
        self.rl_change_epsilon(1)
        #Delete the task
        self.rl_change_task(reward_states=[], rewards=[])
        #Turn off replay
        self.agent.replay = False
        
        #Start the interaction
        reward, state, action, _ = self.rl_start()
        
        #Explore the maze
        for step in range(num_steps):
            reward, state, action, _ = self.rl_step(reward, state, action)
            if self.agent.replay:
                self.agent.memory[(last_state, last_action)] = (reward, state)

            
        
        #Reset the parameters
        self.rl_change_epsilon(old_epsilon)
        self.rl_change_task(reward_states=old_reward_states, rewards=old_rewards)
        self.agent.replay = old_replay
        
        
        
        pass
    
    
    def learn_offline(self, num_replay_steps):
        for step in range(num_replay_steps):
            (last_state, last_action), (reward, state) = random.choice(list(self.agent.memory.items()))
            self.agent.replay_update(last_state, last_action, reward, state)
            
            
    def confront_with_final_trajectory(self):
        """
        Confront agent with change in reward states
        """
        terminal_states = self.env.reward_states
        
        
        for i, terminal_state in enumerate(terminal_states):
            #Get new reward 
            reward = self.env.rewards[i]
            
            #Get adjacent states
            above = terminal_state+self.env.width
            right = terminal_state + 1
            below = terminal_state - self.env.width
            left = terminal_state - 1
            
            
            for transition in [(above, 2), (right, 3), (below, 0), (left, 1)]:
                last_state, last_action = transition
                if transition in self.agent.memory:
                    (reward, state) = self.agent.memory[transition]
                    self.agent.replay_update(last_state, last_action, reward, state)
            
        pass
    
    
    def confront_with_wall(self, new_wall):
        above = new_wall+self.env.width
        right = new_wall + 1
        below = new_wall - self.env.width
        left = new_wall - 1
        
        if above not in self.env.walls:
            self.agent.memory[above, 2] = (-1, above)
            self.agent.replay_forgetting(above, 2, -1, above)
            
        if right not in self.env.walls:
            self.agent.memory[right, 3] = (-1, right)
            self.agent.replay_forgetting(right, 3, -1, right)
        
        if below not in self.env.walls:
            self.agent.memory[below, 0] = (-1, below)
            self.agent.replay_forgetting(below, 0, -1, below)
        
        if left not in self.env.walls:
            self.agent.memory[left, 1] = (-1, left)
            self.agent.replay_forgetting(left, 1, -1, left)
            
            
        pass
        
        
        
        
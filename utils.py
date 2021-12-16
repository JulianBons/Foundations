#This module defines utility function needed for the other modules

import numpy as np
import matplotlib.pyplot as plt


def argmax(q_values):
    """
    Returns the action with the highest q-value. 
    If multiple values share the highest value, randomly sample one. 
    """
    return np.random.choice(np.flatnonzero(q_values == q_values.max()))


def onehot(num_states, state):
    """
    One hot encodes the given state. 
    """
    vec = np.zeros([num_states])
    
    vec[state] = 1
   
    return vec


def matrix_to_ndarray(state, width):
    """
    Turns state from tuple to ndarray
    Arguments:
        state -- tuple, row and column of state
    Output:
        state -- index of ndarray
    """

    row, col = state

    return row*width + col
    
    
    
def ndarray_to_matrix(state, width):
    """
    Turns state from ndarray to tuple
    Arguments:
        state -- index of ndarray
    Output:
        state -- tuple, row and column of state
    """

    return (state//width, state%width)



def plot(size, walls, start_state=None, reward_states=None, current_state=None):
        """
        Plot the current maze
        """
        #Set up maze
        maze = np.zeros(size)
        for loc in walls:
            maze[loc] = -1
            
        #Plot the start_state
        if start_state:
            maze[start_state] = 2
        
        #Plot the reward_states
        if reward_states:
            for loc in reward_states:
                maze[loc] = 1
        
        #Plot the maze
        plt.figure(figsize=(12, 12))
        plt.imshow(maze, origin="lower")
        
        #Plot the current_state
        if current_state:
            plt.text(*reversed(current_state),"A", 
                     ha="center", 
                     va="center", 
                     color="red", 
                     fontsize=25,
                     fontweight="bold",
                     bbox={'facecolor': 'white','boxstyle': 'Round4','pad':0.75})
        
        if start_state:
            plt.text(*reversed(start_state), "S", ha="center", va="center", color="red", fontsize=20, alpha=0.75)
            
        if reward_states:
            for loc in reward_states:
                plt.text(*reversed(loc), "R", ha="center", va="center", color="red", fontsize=20, alpha=0.75)
        
        #plt.show()
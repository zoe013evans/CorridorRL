import numpy as np 
import gym 
import matplotlib.pyplot as plt
import pandas as pd
import pdb 
import random 



def utili_GGF(reward_vector):
    D = len(reward_vector)
    weights = np.zeros(D)
    for w in range(0,D): 
        weights[w] = 1/D
    
    vectors = np.sort(reward_vector)
    final_reward = sum(weights*vectors)
    return(final_reward)


def maxmin1_GGF(reward_vector):
    D = len(reward_vector)
    weights = np.zeros(D)
    weights[0] = 1
    
    vectors = np.sort(reward_vector)
    final_reward = sum(weights*vectors)
    return(final_reward)



v = [1,0,0,0]
reward = utili_GGF(v)



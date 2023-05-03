from resturantGridWorld import resturantGridWorld
import numpy as np 
import gym 
import matplotlib.pyplot as plt
import pandas as pd
import pdb 
import random 
from Q_Learning_Agent_New import train_agent




test_proportions = np.zeros((7,3))

test_proportions[0,:] = [1,0,0] #This is only 1 human jump
test_proportions[1,:] = [0,1,0] #This is only 2 human jump 
test_proportions[2,:] = [0,0,1] #This is only 3 human jump
test_proportions[3,:] = [0.1,0,1] # 10% 1, 90% 3
test_proportions[4,:] = [0.9,0,1] #90% 1 10% 3
test_proportions[5,:] = [0.3,0.6,0.9] #30% 1, 30% 2, 30% 3
test_proportions[6,:] = [0.1,0.9,1] #10% 1, 80% 2, 10% 3
test_proportions[0,:] = [1,0,0] #This is only 1 human jump


for i in range(0,6):
    print("TRAINING SET: ", i)
    occ1, occ2, occ3 = test_proportions[i,:]
    train_agent(occ1, occ2, occ3)




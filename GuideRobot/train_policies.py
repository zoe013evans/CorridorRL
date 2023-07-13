from guideRobotEnv import guideRobot
from Q_Learning_Agent import train
import numpy as np 
import gym 
import matplotlib.pyplot as plt
import pandas as pd
import pdb 
import random 


test_proportions = np.zeros((7,3))
test_proportions[0,:] = [0.1,0.9,1] #10 1, 80% 2, 90% 3 
test_proportions[1,:] = [0,1,0] #This is only 2 human jump 
test_proportions[2,:] = [0,0,1] #This is only 3 human jumppandas
test_proportions[3,:] = [0.1,0,1] #10% 1 90%3 
test_proportions[4,:] = [0.9,0,1] #90% 1, 10% 3
test_proportions[5,:] = [0.33,0.66,1] #30% 1, 30% 2, 30% 3
test_proportions[6,:] = [1,0,0] #This is only 1 human jump

num_policies = len(test_proportions)

for i in range(0,num_policies):
    print(i) 
    train(test_proportions[i,:])

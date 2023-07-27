from guideRobotEnv import guideRobot
import numpy as np 
import gym 


# Load environment and Q table

env = guideRobot(render_mode="human")
episodes = 10


for episode in range(1,episodes):
    obs, info = env.reset()
    done = False 

    while not done: 
        action = env.action_space.sample()
        obs, reward, done, f, info = env.step(action)
        score =+ reward


from guideRobotEnv import guideRobot
import numpy as np 
import gym 
import matplotlib.pyplot as plt
import pandas as pd
import pdb 
import random



def evaluate_agent(Q_table_filename, human_jump, episode_num):
    
    #Make Environment: 
    eval_env = guideRobot(human_jumps=human_jump,render_mode="human")
    eval_env.training_mode = False
    n_obs = eval_env.size 
    n_actions = eval_env.action_space.n

    #Set up Q Table from environment:
    df = pd.read_csv(Q_table_filename)
    Q_table = df.to_numpy()
    Q_table = Q_table.reshape(n_actions, n_obs, n_obs)


    

    #Run an episode: 
    eval_eps= episode_num
    rewards = np.zeros(eval_eps)
    
    for eval_ep in range(0,eval_eps):


        #counter 
        counter = 0

        eval_reward = 0
        #print("EVAL_EP: ", eval_ep)
        obs, info = eval_env.reset()
        done = False 

        current_agent_state = 0 
        current_human_state = 0 

        while not done: 

            if (np.all(Q_table[:,current_human_state,current_agent_state])==0):
                action = random.randint(0,n_actions-1)
            else: 
                action = np.argmax(Q_table[:,current_human_state,current_agent_state])


            obs, reward, done, f, info = eval_env.step(action)
            #print("reward: ", reward)
            eval_reward = eval_reward + reward
            #print("cummulative_Reward: ", eval_reward)

            

            [agent_next_state, filler] = obs.get("agent")
            [human_next_state, filler] = obs.get("human")

            current_agent_state = agent_next_state
            current_human_state = human_next_state

            counter = counter + 1
            #print("counter ", counter)

            if (counter > 50):
                done = True


        rewards[eval_ep] = eval_reward

        #print("EVAL_EP REWARD ", eval_ep, ": ", eval_reward)

    return rewards


    



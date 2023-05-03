from resturantGridWorld import resturantGridWorld
import numpy as np 
import gym 
import matplotlib.pyplot as plt
import pandas as pd
import pdb 
import random 


h_jumps = 3

occ1 = 0.3
occ2 = 0
occ3 = 1


def train_agent(occ1, occ2, occ3):
    exp_filename = "Q_Tables/" + str(occ1) + "_" + str(occ2) + "_" + str(occ3) + ".csv"

    env = resturantGridWorld()

    episodes = 25
    exploration_prob = 1 
    min_explore_prob = 0.01 
    exploration_decreasing_decay = 0.999
    gamma = 0.99
    lr = 0.5

    n_obs = env.size 
    n_actions = env.action_space.n

    #Variables for plotting 
    #Steps:
    step_counter = 0
    total_step_counter = np.array([0])

    #Jumps:
    jumps = np.zeros((episodes,3))

    #Episodes:
    episode_counter = np.array([0])

    #Q_values: 
    initial_Q_values = np.zeros((episodes, n_actions))
    pen_Q_values = np.zeros((episodes, n_actions))

    #Make a Q table that represents all the states: 
    Q_table = np.zeros((n_actions, n_obs, n_obs))
    explored_states = np.zeros((n_obs,n_obs))



    #Learning 

    for episode in range(episodes):
        print("EPISODE: ", episode)

        obs, info = env.reset()

        rand = random.random()

        if (0< rand <= occ1): 
            env._jump_size = 1
        elif(occ1 < rand <=occ2):
            env._jump_size = 2 
        elif(occ2 < rand <= occ3):
            env._jump_size = 3
    

        
        done = False 
        score = 0 
        current_agent_state = 0
        current_human_state = 0
        step_counter = 0  

        one = 0 
        two = 0 
        three = 0 


        while not done: 


            #Exploration vs Explotation
            if np.random.uniform(0,1)<exploration_prob: 
                action = env.action_space.sample()
            else: 
                action = np.argmax(Q_table[:,current_human_state,current_agent_state])

            #Counting Actions:
            if (action == 0 | action == 1): 
                one = one + 1
            if (action == 2 | action == 3): 
                two = two + 1
            if (action == 4 | action == 5):
                three = three + 1


            explored_states[current_human_state,current_agent_state] = 1
            
            

            #Taking Step: 
            obs, reward, done, f, info = env.step(action)
            [agent_next_state, filler] = obs.get("agent")
            [human_next_state, filler] = obs.get("human")


            #Updating Q_Table: 
            # Q_table[action, human_state, agent_state] 

            Q = Q_table[action, current_human_state, current_agent_state]
            Q_table[action, current_human_state, current_agent_state] = Q + lr*(reward + gamma*max(Q_table[:,human_next_state,agent_next_state])-Q)

            #Checking if done: 
            if (done == False):
                current_agent_state = agent_next_state
                current_human_state = human_next_state

    
            step_counter = step_counter + 1 


            #if (step_counter>50):
            #   done = True


            new_exploration_prob = exploration_prob * exploration_decreasing_decay
            if new_exploration_prob > min_explore_prob: 
                exploration_prob = new_exploration_prob
            final_table = Q_table 


        
        # Episode has finished
        # Now we plot information: 


        #Jump per episode: 
        jumps[episode,0] = one
        jumps[episode,1] = two
        jumps[episode,2] = three
        data = pd.DataFrame(jumps)
        data.columns = ['one', 'two', 'three']
        data.to_csv("saved_tables/jumps_per_episode.csv", index=None)


        #Steps per episode: 
        total_step_counter = np.append(total_step_counter, [step_counter])
        episode_counter = np.append(episode_counter, episode+1)
        pd.DataFrame(total_step_counter).to_csv("saved_tables/step_results.csv", header=None,index=None)


        #Initial State 
        #When both human and agent are in state 0 

        initial_state = Q_table[:,0,0]
        initial_Q_values[episode,:] = initial_state
        data = pd.DataFrame(initial_Q_values)
        data.columns = ['right 1', 'left 1', 'right 2', 'left 2', 'right 3','left 3']
        data.to_csv("saved_tables/initial_Q_results.csv", index=None)
        
        #Final State 
        #When the agent is in state n and the human is in state n-1 
        #Hypothetically 

        pen_state = Q_table[:,n_obs-2, n_obs-1]
        pen_Q_values[episode, :] = pen_state
        data = pd.DataFrame(pen_Q_values)
        data.columns = ['right 1', 'left 1', 'right 2', 'left 2', 'right 3','left 3']
        data.to_csv("saved_tables/pen_Q_results.csv", index=None)


    final_table = Q_table


    #Reshape from 3D to 2D 
    final_save_table = final_table.reshape(final_table.shape[0],-1)
    data = pd.DataFrame(final_save_table)
    #Now, we save our final policy: ;
    filename = "saved_tables/finalQTableMixedTest.csv"
    data.to_csv(exp_filename,index=None)




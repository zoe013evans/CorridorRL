from guideRobotEnv import guideRobot
import numpy as np 
import gym 
import matplotlib.pyplot as plt
import pandas as pd
import pdb 
import random 

def train(human_f_prop):

    #Gathering the amounts of different humans robot will see: 
    occ1, occ2, occ3 = human_f_prop 

    #Setting up file name for saving Q_table:
    exp_filename = "Q_Tables/Q_table" + str(occ1) + "_" + str(occ2) + "_" + str(occ3) + ".csv"

    #Creating environment: 
    env = guideRobot()
    env.training_mode = True

    #Setting up parameters:
    episodes = 50000
    exploration_prob = 1 
    min_explore_prob = 0.01 
    exploration_decreasing_decay = 0.99999
    gamma = 0.99
    lr = 0.5

    #Gathering size of observation and actions: 
    n_obs = env.size 
    n_actions = env.action_space.n


    #Gathering variables for plotting graphs: 
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



    #Make a Q table that represents the values of all state action pairs: 
    Q_table = np.zeros((n_actions, n_obs, n_obs))
    explored_states = np.zeros((n_obs,n_obs))


    #Learning 
    for episode in range(episodes):

        obs, info = env.reset()
        rand = random.random()

        #Pick which human the robot will see this episode: 
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
                if (np.all(Q_table[:,current_human_state,current_agent_state])==0):
                    action = random.randint(0,n_actions-1)
                else: 
                    action = np.argmax(Q_table[:,current_human_state,current_agent_state])


            #Counting Actions for plotting:
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
            Q = Q_table[action, current_human_state, current_agent_state]
            Q_table[action, current_human_state, current_agent_state] = Q + lr*(reward + gamma*max(Q_table[:,human_next_state,agent_next_state])-Q)

            #Checking if done: 
            if (done == False):
                current_agent_state = agent_next_state
                current_human_state = human_next_state

    
            step_counter = step_counter + 1 

            new_exploration_prob = exploration_prob * exploration_decreasing_decay
            if new_exploration_prob > min_explore_prob: 
                exploration_prob = new_exploration_prob

            #Saving final Q_table     
            final_table = Q_table 

        # Episode has finished
        # Now we plot information: 

        #Jump per episode: 
        jumps[episode,0] = one
        jumps[episode,1] = two
        jumps[episode,2] = three
        data = pd.DataFrame(jumps)
        data.columns = ['one', 'two', 'three']
        filen = "saved_tables/jumps_per_episode" + str(occ1) + "_" + str(occ2) + "_" + str(occ3) + ".csv"
        data.to_csv(filen, index=None)


        #Steps per episode: 
        total_step_counter = np.append(total_step_counter, [step_counter])
        episode_counter = np.append(episode_counter, episode+1)
        filen = "saved_tables/step_results.csv" + str(occ1) + "_" + str(occ2) + "_" + str(occ3) + ".csv"
        pd.DataFrame(total_step_counter).to_csv(filen, header=None,index=None)


        #Initial State 
        #When both human and agent are in state 0 

        initial_state = Q_table[:,0,0]
        initial_Q_values[episode,:] = initial_state
        data = pd.DataFrame(initial_Q_values)
        data.columns = ['right 1', 'left 1', 'right 2', 'left 2', 'right 3','left 3']
        filen = "saved_tables/initial_Q_results" + str(occ1) + "_" + str(occ2) + "_" + str(occ3) + ".csv"
        data.to_csv(filen, index=None)
        
        #Final State 
        #When the agent is in state n and the human is in state n-1 
        #Hypothetically 

        pen_state = Q_table[:,n_obs-2, n_obs-1]
        pen_Q_values[episode, :] = pen_state
        data = pd.DataFrame(pen_Q_values)
        data.columns = ['right 1', 'left 1', 'right 2', 'left 2', 'right 3','left 3']
        filen = "saved_tables/pen_Q_results" + str(occ1) + "_" + str(occ2) + "_" + str(occ3) + ".csv"
        data.to_csv(filen, index=None)


    final_table = Q_table

    #Reshape from 3D to 2D 
    final_save_table = final_table.reshape(final_table.shape[0],-1)
    data = pd.DataFrame(final_save_table)


    states = []
    #easy_Q_table = zeros(n_actions+1,n_obs)
    for n in range(0,n_obs):
        for n2 in range(0,n_obs):
            state = (n,n2)
            states.append(state)

    actions = np.zeros((len(states),n_actions))
    for n in range(0,len(states)):
        h,a = states[n]
        action = final_table[:, h, a]
        actions[n,:] = action



    df_Q_table = pd.DataFrame({'states': states,
                        'right 1': actions[:,0],
                        'left 1': actions[:,1],
                        'right 2': actions[:,2],
                        'left 2': actions[:,3],
                        'right 3': actions[:,4],
                        'left 3': actions[:,5]})


    filen = "saved_tables/Q_table_orginal" + str(occ1) + "_" + str(occ2) + "_" + str(occ3) + ".csv"

    df_Q_table.to_csv(filen, index=None)
    
    #Now, we save our final policy: ;
    data.to_csv(exp_filename,index=None)




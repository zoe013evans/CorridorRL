from guideRobotEnv import guideRobot
import numpy as np 
import gym 
import matplotlib.pyplot as plt
import pandas as pd
import pdb 
import random 



def Q_table_2_readable_csv(final_table, env, path_name):
    n_obs = env.size
    n_actions = env.action_space.n
    states = []
    #easy_Q_table = zeros(n_actions+1,n_obs)
    for n in range(0,n_obs):
        for n2 in range(0,n_obs):
            state = (n,n2)
            states.append(state)


    actionsf = np.zeros((len(states),n_actions))
    for n in range(0,len(states)):
        h,a = states[n]
      #  print("state: ", (h,a))
        action = final_table[:, h, a]
       # print("action: ", action)
        actionsf[n,:] = action
        #print("action f: ", actionsf[n,:])

    df_fair_table = pd.DataFrame({'states': states,
                        'right 1': actionsf[:,0],
                        'left 1': actionsf[:,1],
                        'right 2': actionsf[:,2],
                        'left 2': actionsf[:,3],
                        'right 3': actionsf[:,4],
                        'left 3': actionsf[:,5]})


    
    df_fair_table.to_csv(path_name, index=None)
    
def save_step_plot(total_step_counter,step_counter,episode,file_name):
    if episode % 1000 == 999:
        total_step_counter = np.append(total_step_counter, step_counter)
        pd.DataFrame(total_step_counter).to_csv(file_name, header=None,index=None)    
    return total_step_counter
    

def save_initial_state_plot(Q_table,initial_Q_values,episode,file_name):
    index = int(episode/1000)
    if episode % 1000 == 999:
        initial_state = Q_table[:,0,0]
        initial_Q_values[index,:] = initial_state
        data = pd.DataFrame(initial_Q_values)
        data.columns = ['right 1', 'left 1', 'right 2', 'left 2', 'right 3','left 3']
        data.to_csv(file_name, index=None)
    return initial_Q_values

def save_pen_state_plot(Q_table,n_obs,pen_Q_values,episode,file_name):
    #index = int(episode/1000)
    index = int(episode/1000)
    if episode % 1000 == 999: 
        pen_state = Q_table[:,n_obs-2, n_obs-1]
        pen_Q_values[index, :] = pen_state
        data = pd.DataFrame(pen_Q_values)
        data.columns = ['right 1', 'left 1', 'right 2', 'left 2', 'right 3','left 3']
        data.to_csv(file_name, index=None)
    return pen_Q_values


def train(human_f_prop):

    print("Q Learning Original")

    #Gathering the amounts of different humans robot will see: 
    occ1, occ2, occ3 = human_f_prop 

    Q_learning_type = 'Q_Learning_Original/'
    policy_type = str(occ1) + "_" + str(occ2) + "_" + str(occ3)

    #Setting up file name for saving Q_table:
    exp_filename = "Q_Tables/" + Q_learning_type + "Q_table_" + policy_type + ".csv"
    plot_save_location = "saved_tables/" + Q_learning_type

    #Creating environment: 
    env = guideRobot()
    env.training_mode = True

    #Setting up parameters:
    episodes = 500000
    exploration_prob = 1 
    min_explore_prob = 0.01 
    exploration_decreasing_decay = 0.9999
    gamma = 0.99
    lr = 0.1

    #Gathering size of observation and actions: 
    n_obs = env.size 
    n_actions = env.action_space.n

    plot_rate = 1000
    #Gathering variables for plotting graphs: 
    #Steps:
    step_counter = 0
    total_step_counter = np.array([0])
    #Jumps:
    jumps = np.zeros((int(episodes/plot_rate),3))
    #Q_values: 
    initial_Q_values = np.zeros((500, n_actions))
    pen_Q_values = np.zeros((500, n_actions))



    #Make a Q table that represents the values of all state action pairs: 
    Q_table = np.zeros((n_actions, n_obs, n_obs))

    feature = 0
    explored_states = np.zeros((n_obs,n_obs))
    print(explored_states)

    #Learning 
    for episode in range(episodes):
        obs, info = env.reset()
        rand = random.random()

        print("episode: ", episode)

        #Pick which human the robot will see this episode: 
        if (0< rand <= occ1): 
            env._jump_size = 1
            feature = 1
        elif(occ1 < rand <=occ2):
            env._jump_size = 2 
            feature = 2
        elif(occ2 < rand <= occ3):
            env._jump_size = 3
            feature = 3
    
        
        #print("feature: ", feature)
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

            #print("(Current human, current agent)", (current_human_state,current_agent_state))
            explored_states[current_human_state,current_agent_state] = 1

            if ((explored_states[19,1]) == 1):
                pdb.set_trace()
            
            #Taking Step: 
            obs, reward, done, f, info = env.step(action)
            [agent_next_state, filler] = obs.get("agent")
            [human_next_state, filler] = obs.get("human")

            

            if (current_human_state==((env.size)-1)):
                Q_table[action,current_human_state,current_agent_state] = 0
            else: 
                #Updating Q_Table:   
                Q = Q_table[action, current_human_state, current_agent_state]
                temporal_difference = reward + (gamma*np.max(Q_table[:,human_next_state,agent_next_state])) - Q
                new_qsa = Q + (lr*temporal_difference)
                Q_table[action,current_human_state,current_agent_state] = new_qsa

            step_counter = step_counter + 1 


            

            #Checking if done: 
            if (done == False):
                current_agent_state = agent_next_state
                current_human_state = human_next_state
            

            

            new_exploration_prob = exploration_prob * exploration_decreasing_decay
            if new_exploration_prob > min_explore_prob: 
                exploration_prob = new_exploration_prob

            #Saving final Q_table     
            final_table = Q_table 

        # Episode has finished
        # Now we plot information: 

        file_name = "saved_tables/Q_Learning_Original" + "/step_results_" + policy_type + ".csv"
        total_step_counter = save_step_plot(total_step_counter,step_counter,episode,file_name)
        
        file_name = "saved_tables/Q_Learning_Original" + "/initial_Q_values_" + policy_type + ".csv"
        initial_Q_values = save_initial_state_plot(Q_table,initial_Q_values,episode,file_name)

        file_name = "saved_tables/Q_Learning_Original" + "/pen_Q_values_" + policy_type + ".csv"
        pen_Q_values = save_pen_state_plot(Q_table,n_obs,pen_Q_values,episode,file_name)

        
    final_table = Q_table

    #Reshape from 3D to 2D 
    final_save_table = final_table.reshape(final_table.shape[0],-1)
    data = pd.DataFrame(final_save_table)

    path_name = "Clear_Q_tables/" + Q_learning_type + "Q_table" + str(occ1) + "_" + str(occ2) + "_" + str(occ3) + ".csv"
    Q_table_2_readable_csv(final_table, env, path_name)
    
    #Now, we save our final policy: 
    data.to_csv(exp_filename,index=None)





print("train: 1,0,0")
train([1,0,0])
print("train: 0.1,0.9,1")
train([0.1,0.9,1]) #10 1, 80% 2, 90% 3 
print("train: 0,1,0")
train([0,1,0]) #This is only 2 human jump 
print("train: 0,0,1")
train([0,0,1]) #This is only 3 human jumppandas
print("train: 0.1,0,1")
train([0.1,0,1]) #10% 1 90%3 
print("train: 0.9,0,1")
train([0.9,0,1]) #90% 1, 10% 3
print("train: 0.33,0.66,1")
train([0.33,0.66,1]) #30% 1, 30% 2, 30% 3

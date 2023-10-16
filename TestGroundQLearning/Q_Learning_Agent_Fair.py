from guideRobotEnv import guideRobot
import numpy as np 
import gym 
import matplotlib.pyplot as plt
import pandas as pd
import pdb 
import random 
from torch.utils.tensorboard import SummaryWriter
from matplotlib.animation import FuncAnimation


def action_selection_minmax(Q_table_fair,current_human_state,current_agent_state,exploration_prob,env): 
    #We pass the Q_table to this function and use it to select actions 

    n_actions = env.action_space.n
    Q_table_add_1_2 = np.add(Q_table_fair[:,current_human_state,current_agent_state,1],Q_table_fair[:,current_human_state,current_agent_state,2])
    Q_table_fair_temp = np.add((Q_table_fair[:,current_human_state,current_agent_state,0]),Q_table_add_1_2)

    #Exploration vs Explotation
    if np.random.uniform(0,1)<exploration_prob: 
        action = env.action_space.sample()
    else: 
        if (np.all(Q_table_fair_temp)==0):
            action = env.action_space.sample()
        else: 
            action = np.argmax(Q_table_fair_temp)
    
    return action 


def action_selection_argmax(Q_table_fair,current_human_s,current_agent_s,feat,explore_prob,env): 
    #We pass the Q_table to this function and use it to select actions 

    n_actions = env.action_space.n

    #Exploration vs Explotation
    if np.random.uniform(0,1)<explore_prob: 
        action = env.action_space.sample()
    else: 
        if (np.all(Q_table_fair[:,current_human_s,current_agent_s,feat])==0):
            action = random.randint(0,n_actions-1)
        else: 
            action = np.argmax(Q_table_fair[:,current_human_s,current_agent_s,feat])
    
    return action 


def save_step_plot(total_step_counter,step_counter,episode,file_name):
    if episode % 1000 == 999:
        total_step_counter = np.append(total_step_counter, step_counter)
        pd.DataFrame(total_step_counter).to_csv(file_name, header=None,index=None)    
    return total_step_counter


def save_initial_state_plot(Q_table_fair,initial_Q_values_f0, initial_Q_values_f1, initial_Q_values_f2,episode,policy_type,fair_type):
    index = int(episode/1000)
    if episode % 1000 == 999:
        initial_state_f0 = Q_table_fair[:,0,0,0]
        initial_state_f1 = Q_table_fair[:,0,0,1]
        initial_state_f2 = Q_table_fair[:,0,0,2]
        initial_Q_values_f0[index,:] = initial_state_f0
        initial_Q_values_f1[index,:] = initial_state_f1
        initial_Q_values_f2[index,:] = initial_state_f2

        data0 = pd.DataFrame(initial_Q_values_f0)
        data1 = pd.DataFrame(initial_Q_values_f1)
        data2 = pd.DataFrame(initial_Q_values_f2)

        file_name0 = "saved_tables/Q_Learning_" + fair_type + "/initial_Q_values_f0" + policy_type + ".csv"
        file_name1 = "saved_tables/Q_Learning_" + fair_type +"/initial_Q_values_f1" + policy_type + ".csv"
        file_name2 = "saved_tables/Q_Learning_" + fair_type +"/initial_Q_values_f2" + policy_type + ".csv"


        data0.columns = ['right 1', 'left 1', 'right 2', 'left 2', 'right 3','left 3']
        data1.columns = ['right 1', 'left 1', 'right 2', 'left 2', 'right 3','left 3']
        data2.columns = ['right 1', 'left 1', 'right 2', 'left 2', 'right 3','left 3']


        data0.to_csv(file_name0, index=None)
        data1.to_csv(file_name1, index=None)
        data2.to_csv(file_name2, index=None)

    return initial_Q_values_f0, initial_Q_values_f1, initial_Q_values_f2

def save_pen_state_plot(pen_Q_values,episode,Q_table,occ1,occ2,occ3,n_obs):
    index = int(episode/1000)

    if episode % 1000 == 999:
        pen_state = Q_table[:,n_obs-2, n_obs-1]
        pen_Q_values[index, :] = pen_state
        data = pd.DataFrame(pen_Q_values)
        data.columns = ['right 1', 'left 1', 'right 2', 'left 2', 'right 3','left 3']
        filen = "saved_tables_fair/pen_Q_results" + str(occ1) + "_" + str(occ2) + "_" + str(occ3) + ".csv"
        data.to_csv(filen, index=None)
    return pen_Q_values


def save_final_Q_table(Q_table,exp_filename):
    final_save_table = Q_table.reshape(Q_table.shape[0],-1)
    data = pd.DataFrame(final_save_table)
    data.to_csv(exp_filename,index=None)

def Q_fair_2_Q_normal(Q_table_fair, env, n_features, occ1, occ2, occ3):
    #How we convert a Q table that includes features into a Q table that is standard 
    n_obs = env.size 
    n_actions = env.action_space.n    
    print(str(occ1) + "_" + str(occ2) + "_" + str(occ3))
    f0_prop = occ1 
    if (occ2 == 0): 
        f1_prop = 0 
    else: 
        f1_prop = occ2 - occ1

    if (occ3 == 0):
        f2_prop = 0 
    else: 
        f2_prop = float(occ3) - occ2 

    print("f0_prop: ", f0_prop)
    print("f1_prop: ", f1_prop)
    print("f2_prop: ", f2_prop)

    Q_table = np.zeros((n_actions,n_obs,n_obs))

    occ1 = f0_prop
    occ2 = f1_prop
    occ3 = f2_prop 

    for action in range(0,n_actions):
            for n_obs1 in range(0,n_obs):
                for n_obs2 in range(0,n_obs): 
                    Q_table[action,n_obs1,n_obs2] = occ1*Q_table_fair[action,n_obs1,n_obs2,0] + occ2*Q_table_fair[action,n_obs1,n_obs2,1] + occ3*Q_table_fair[action,n_obs1,n_obs2,2]
    return Q_table


def Q_fair_2_Q_normal_minmax(Q_table_fair, env, n_features, debug_file_name):
    n_obs = env.size 
    n_actions = env.action_space.n
    final_fair_table_minimax = np.zeros((n_actions,n_obs,n_obs))
    for s_h in range(0,n_obs):
        for s_a in range(0,n_obs):

            dump2textfile = []

            state_st = str((s_h,s_a))
            a0 = np.min(Q_table_fair[0,s_h,s_a,:])
            a1 = np.min(Q_table_fair[1,s_h,s_a,:])
            a2 = np.min(Q_table_fair[2,s_h,s_a,:])
            a3 = np.min(Q_table_fair[3,s_h,s_a,:])
            a4 = np.min(Q_table_fair[4,s_h,s_a,:])
            a5 = np.min(Q_table_fair[5,s_h,s_a,:])

            Action_values_f0 = Q_table_fair[:,s_h,s_a,0]
            Action_values_f1 = Q_table_fair[:,s_h,s_a,1]
            Action_values_f2 = Q_table_fair[:,s_h,s_a,2]

            st_a0 = "Action_values_f0: " + str(Action_values_f0)
            st_a1 = "Action_values_f1: " + str(Action_values_f1)
            st_a2 = "Action_values_f2: " + str(Action_values_f2)


            min_action_values = [a0,a1,a2,a3,a4,a5]
            st_min_vals = "Min Values: " + str(min_action_values)
            to_file = [state_st, st_a0,st_a1,st_a2, st_min_vals]

            with open(debug_file_name,'a') as f: 
                f.write('\n'.join(to_file))
                f.write('\n')
                f.write(' ')
                f.write('\n')
            
            final_fair_table_minimax[:,s_h,s_a] = min_action_values

    return final_fair_table_minimax

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
        action = final_table[:, h, a]
        actionsf[n,:] = action


    df_fair_table = pd.DataFrame({'states': states,
                        'right 1': actionsf[:,0],
                        'left 1': actionsf[:,1],
                        'right 2': actionsf[:,2],
                        'left 2': actionsf[:,3],
                        'right 3': actionsf[:,4],
                        'left 3': actionsf[:,5]})


    
    df_fair_table.to_csv(path_name, index=None)
    

def Fair_train(human_f_prop,fair_type):

    #Gathering the amounts of different humans robot will see: 
    occ1, occ2, occ3 = human_f_prop 
    print(len(human_f_prop))

    
    #Setting up file name for saving Q_table:
    policy_type = str(occ1) + "_" + str(occ2) + "_" + str(occ3)

    exp_filename = "Q_Tables/Q_Learning_" + fair_type + "/Q_table_" + policy_type + ".csv"

    #Creating environment: 
    env = guideRobot()
    env.training_mode = True

    #Setting up parameters:
    episodes = 300000
    exploration_prob = 1 
    min_explore_prob = 0.01 
    exploration_decreasing_decay = 0.9999
    gamma = 0.99
    lr = 0.1
    plot_rate = 1000

    #Gathering size of observation and actions: 
    n_obs = env.size 
    n_actions = env.action_space.n

    #Gathering variables for plotting graphs: 
    #Steps:
    step_counter = 0
    total_step_counter = np.array([0])
    #Jumps:
    jumps = np.zeros((500,3))
    #Q_values: 
    initial_Q_values_f0 = np.zeros((500, n_actions))
    initial_Q_values_f1 = np.zeros((500, n_actions))
    initial_Q_values_f2 = np.zeros((500, n_actions))

    pen_Q_values = np.zeros((500, n_actions))

    
    #Make a Q table that represents the values of all state action pairs: 
    Q_table = np.zeros((n_actions, n_obs, n_obs))
    explored_states = np.zeros((n_obs,n_obs))


    n_features = len(human_f_prop)
    Q_table_fair = np.zeros((n_actions,n_obs,n_obs,n_features))


    #Learning 
    for episode in range(episodes):
        print("episode: ", episode)

        obs, info = env.reset()
        feature = 0

        rand = random.random()

        #Pick which human the robot will see this episode: 
        if (0< rand <= occ1): 
            env._jump_size = 1
            feature = 0
        elif(occ1 < rand <=occ2):
            env._jump_size = 2 
            feature = 1
        elif(occ2 < rand <= occ3):
            env._jump_size = 3
            feature = 2    

        done = False 
        score = 0 
        current_agent_state = 0
        current_human_state = 0
        step_counter = 0  
        
        while not done: 
            
            action = action_selection_argmax(Q_table_fair,current_human_state,current_agent_state,feature,exploration_prob,env)
       
            
            #Taking Step: 
            obs, reward, done, f, info = env.step(action)
            [agent_next_state, filler] = obs.get("agent")
            [human_next_state, filler] = obs.get("human")



            if (current_human_state==((env.size)-1)):
                Q_table_fair[action,current_human_state,current_agent_state,feature] = 0
            else: 
                #Updating Feature Q Table: 
                Q_fe = Q_table_fair[action, current_human_state, current_agent_state,feature]
                temporal_difference = reward + (gamma*np.max(Q_table_fair[:,human_next_state,agent_next_state,feature])) - Q_fe
                Q_table_fair[action, current_human_state, current_agent_state,feature] = Q_fe + (lr*temporal_difference)

            #Checking if done: 
            
            if (done == False):
                current_agent_state = agent_next_state
                current_human_state = human_next_state

    
    

            new_exploration_prob = exploration_prob * exploration_decreasing_decay
            if new_exploration_prob > min_explore_prob: 
                exploration_prob = new_exploration_prob

        #Episode Finished
        # Now we plot information: 

        file_name = "saved_tables/Q_Learning_" + fair_type + "/step_results_" + policy_type + ".csv"
        total_step_counter = save_step_plot(total_step_counter,step_counter,episode,file_name)
        initial_Q_values_f0, initial_Q_values_f1, initial_Q_values_f2 = save_initial_state_plot(Q_table_fair,initial_Q_values_f0, initial_Q_values_f1, initial_Q_values_f2,episode,policy_type,fair_type)
    #Turn Feature-State Q table into just state Q table: 

    if (fair_type == "Normal"): 
        final_table = Q_fair_2_Q_normal(Q_table_fair, env, n_features, occ1, occ2, occ3)
    elif(fair_type == "Minmax"): 
        debug_name = str(occ1) + '_' + str(occ2) + '_' + str(occ3) + '_debug.csv'
        final_table = Q_fair_2_Q_normal_minmax(Q_table_fair, env, n_features,debug_name)
    

    path_name = "Clear_Q_tables/Q_Learning_" + fair_type + "/Q_Table" + str(occ1) + "_" + str(occ2) + "_" + str(occ3) + ".csv"

    Q_table_2_readable_csv(final_table, env, path_name)

    #Reshape from 3D to 2D 
    save_final_Q_table(final_table,exp_filename)
   
print("train: 1,0,0")
Fair_train([1,0,0],"Minmax") 
print("train: 0.1,0.9,1")
Fair_train([0.1,0.9,1], "Minmax") #10 1, 80% 2, 90% 3  """
print("train: 0,1,0")
Fair_train([0,1,0], "Minmax") #This is only 2 human jump 
print("train: 0,0,1")
Fair_train([0,0,1], "Minmax") #This is only 3 human jumppandas
print("train: 0.1,0,1")
Fair_train([0.1,0,1], "Minmax") #10% 1 90%3 
print("train: 0.9,0,1")
Fair_train([0.9,0,1], "Minmax") #90% 1, 10% 3
print("train: 0.33,0.66,1")
Fair_train([0.33,0.66,1],"Minmax") #30% 1, 30% 2, 30% 3





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
    print(len(human_f_prop))

    #Setting up file name for saving Q_table:
    exp_filename = "Q_Tables_Fair/Q_table" + str(occ1) + "_" + str(occ2) + "_" + str(occ3) + ".csv"

    #Creating environment: 
    env = guideRobot()
    env.training_mode = True

    #Setting up parameters:
    episodes = 1000
    exploration_prob = 1 
    min_explore_prob = 0.01 
    exploration_decreasing_decay = 0.999
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


    n_features = len(human_f_prop)
    Q_table_fair = np.zeros((n_actions,n_obs,n_obs,n_features))

    Q_table_fair2 = np.zeros((n_actions,n_obs,n_obs))





    #Learning 
    for episode in range(episodes):
        print("episode: ", episode)

        obs, info = env.reset()
        rand = random.random()

        feature = 0

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

        one = 0 
        two = 0 
        three = 0 


        while not done: 

            #Exploration vs Explotation
            if np.random.uniform(0,1)<exploration_prob: 
                action = env.action_space.sample()
            else: 
                #if (np.all(Q_table[:,current_human_state,current_agent_state])==0):
                    #action = random.randint(0,n_actions-1)
                if (np.all((Q_table_fair[:,human_next_state, agent_next_state,:]))==0):
                    action = random.randint(0,n_actions-1) 
                else: 
                    #action = np.argmax(Q_table[:,current_human_state,current_agent_state])
                    action = np.argmax(np.sum((Q_table[:,human_next_state, agent_next_state,:])))


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
            #Q = Q_table[action, current_human_state, current_agent_state]
            #Q_table[action, current_human_state, current_agent_state] = Q + lr*(reward + gamma*max(Q_table[:,human_next_state,agent_next_state])-Q)


            #Updating Feature Q Table: 
            Q_fe = Q_table_fair[action, current_human_state, current_agent_state,feature]
            #Q_table_fair[action, current_human_state, current_agent_state,feature] = Q_fe + lr*(reward + gamma*max(Q_table[:,human_next_state,agent_next_state])-Q_fe)

            Q_table_fair[action, current_human_state, current_agent_state, feature] = Q_fe + lr*(reward + gamma*max((Q_table_fair[:,human_next_state, agent_next_state,feature]))-Q_fe)

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

        final_fair_table = np.zeros((n_actions, n_obs, n_obs))

        for action in range(0,n_actions):
            for n_obs1 in range(0,n_obs):
                for n_obs2 in range(0,n_obs):
                    final_fair_table[action, n_obs1, n_obs2] = np.sum(Q_table_fair[action,n_obs1, n_obs2,:])

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

    states = []
    #easy_Q_table = zeros(n_actions+1,n_obs)
    for n in range(0,n_obs):
        for n2 in range(0,n_obs):
            state = (n,n2)
            states.append(state)

            
    actions = np.zeros((len(states),n_actions))
    actionsf = np.zeros((len(states), n_actions))
    for n in range(0,len(states)):
        h,a = states[n]
        action = final_table[:, h, a]
        actions[n,:] = action

        actionf = final_fair_table[:,h,a]
        actionsf[n,:] = actionf 



    df_easy_table = pd.DataFrame({'states': states,
                        'right 1': actions[:,0],
                        'left 1': actions[:,1],
                        'right 2': actions[:,2],
                        'left 2': actions[:,3],
                        'right 3': actions[:,4],
                        'left 3': actions[:,5]})

    df_fair_table = pd.DataFrame({'states': states,
                        'right 1': actionsf[:,0],
                        'left 1': actionsf[:,1],
                        'right 2': actionsf[:,2],
                        'left 2': actionsf[:,3],
                        'right 3': actionsf[:,4],
                        'left 3': actionsf[:,5]})


    
    df_easy_table.to_csv("saved_tables/df_Q.csv",index=None)
    df_fair_table.to_csv("saved_tables/df_Q_F.csv", index=None)
    




        
            
    
    
    #Now, we save our final policy: ;
    data.to_csv(exp_filename,index=None)


    final_fair_table = final_fair_table.reshape(final_fair_table.shape[0],-1)
    data = pd.DataFrame(final_fair_table)
    data.to_csv("final_fair_table.csv",index=None)


    pdb.set_trace()
   


train([1,0,0])
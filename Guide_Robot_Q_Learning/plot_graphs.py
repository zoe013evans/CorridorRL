import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
import seaborn as sns
import pdb 


## PLOTTING FROM SAVED FILES 

policy = '1_0_0'
fair_type = 'Leximin'
file_path = 'saved_tables/Q_Learning_' + fair_type + '/' 

filen = file_path + 'step_results_' + policy + '.csv'
df = pd.read_csv(filen)
df.plot()
plt.title('Steps per episode')
plt.show()


## Plot the value function for initial states of episodes 
filen = file_path + 'initial_Q_values_' + policy + '.csv'
df = pd.read_csv(filen)
df.plot()
plt.title("Value Function from Initial States")
plt.show()



# Plot smoothed version of the value function for initial states of episodes 
df_roll = pd.DataFrame()
df_roll['right 1'] = df['right 1'].rolling(3).sum()
df_roll['left 1'] = df['left 1'].rolling(3).sum()
df_roll['right 2'] = df['right 2'].rolling(3).sum()
df_roll['left 2'] = df['left 2'].rolling(3).sum()
df_roll['right 3'] = df['right 3'].rolling(3).sum()
df_roll['left 3'] = df['left 3'].rolling(3).sum()
df_roll.plot()
plt.title("Value Function from Initial States - With Rolling window")
plt.show()



## Plot the value function of second to last states over episodes 
filen = file_path + 'pen_Q_values_' + policy + '.csv'
df = pd.read_csv(filen)
df.plot()
plt.title("Value Function from Penultimate States")
plt.show()


##Plot the smoothed version of the value function for last states over episodes 
df_roll = pd.DataFrame()
df_roll['right 1'] = df['right 1'].rolling(3).sum()
df_roll['left 1'] = df['left 1'].rolling(3).sum()
df_roll['right 2'] = df['right 2'].rolling(3).sum()
df_roll['left 2'] = df['left 2'].rolling(3).sum()
df_roll['right 3'] = df['right 3'].rolling(3).sum()
df_roll['left 3'] = df['left 3'].rolling(3).sum()
df_roll.plot()
plt.title("Value Function from Penultimate States - With Rolling window")
plt.show()


'''
## Plot the average size of jumps over time 
#filen = 'saved_tables_fair_MOMDP/jumps_per_episode' + policy + '.csv'
#df = pd.read_csv(filen)
#df.plot()
#plt.title("Average jump per episode")
#plt.show()


##Plot the average jumps over time 
df_roll = pd.DataFrame()
df_roll['one'] = df['one'].rolling(10).sum()
df_roll['two'] = df['two'].rolling(10).sum()
df_roll['three'] = df['three'].rolling(10).sum()

df_roll.plot()
plt.title("Average jump per episode - With Rolling Window")
plt.show()

'''
from guideRobotEnv import guideRobot
import numpy as np 
import gym 
import matplotlib.pyplot as plt
import pandas as pd
import pdb 
from evaluation import evaluate_agent


folder = 'Q_Tables/Q_Learning_Leximin/Q_table_'
n_runs = 3

#TEST ONE: Evaluating agent on models trained on only one agent: 


print("1")
#HUMAN SPEED 1: 
filename = folder + '1_0_0.csv'
print("SLOW POLICY WITH SLOW PERSON")
one_results1 = evaluate_agent(filename,1,n_runs)

print("SLOW POLICY WITH MEDIUM PERSON")
one_results2 = evaluate_agent(filename, 2, n_runs)

print("SLOW POLICY WITH FAST PERSON")
one_results3 = evaluate_agent(filename,3,n_runs)


#Creating a big table: 

data= {
    'Methods':['1.0_0.0_0.0'],
    'jump 1': [np.mean(one_results1)],
    'jump 2': [np.mean(one_results2)],
    'jump 3': [np.mean(one_results3)]
    
    }

resultsTable = pd.DataFrame(data)


print("2")

#HUMAN SPEED 2: 
filename = folder + '0_1_0.csv'
two_results1 = evaluate_agent(filename,1,n_runs)
two_results2 = evaluate_agent(filename, 2, n_runs)
two_results3 = evaluate_agent(filename,3,n_runs)


nextrow = ['0.0_1.0_0.0',np.mean(two_results1), np.mean(two_results2), np.mean(two_results3)]
resultsTable.loc[len(resultsTable)] = nextrow

print("3")
#HUMAN_SPEED 3: 
filename = folder + '0_0_1.csv'
three_results1 = evaluate_agent(filename,1,n_runs)
three_results2 = evaluate_agent(filename, 2, n_runs)
three_results3 = evaluate_agent(filename,3,n_runs)

nextrow = ['0.0_0.0_1.0',np.mean(three_results1), np.mean(three_results2), np.mean(three_results3)]
resultsTable.loc[len(resultsTable)] = nextrow



#TEST 2: Evaluating models trained on a mix of human speeds with each human speed
#90% 3 and 10% 1 
#evaluate on each, plot the results in a graph: 

filename = folder + '0.1_0_1.csv'
mix1_results1 = evaluate_agent(filename,1,n_runs)
mix1_results2 = evaluate_agent(filename,2,n_runs)
mix1_results3 = evaluate_agent(filename,3,n_runs)

nextrow = ['0.1_0.0_1.0',np.mean(mix1_results1), np.mean(mix1_results2),np.mean(mix1_results3)]
resultsTable.loc[len(resultsTable)] = nextrow


#30% 1, 30% 2 30% 3 
#evaluate on each, plot the results in a graph:
filename = folder + '0.33_0.66_1.csv'
mix2_results1 = evaluate_agent(filename,1,n_runs)
mix2_results2 = evaluate_agent(filename, 2, n_runs)
mix2_results3 = evaluate_agent(filename,3,n_runs)

nextrow = ['0.33_0.66_1.0',np.mean(mix2_results1), np.mean(mix2_results2),np.mean(mix2_results3)]
resultsTable.loc[len(resultsTable)] = nextrow



#10% 1, 80% 2, 10% 3 
#evaluate on each, plot the results in a graph: 

filename = folder + '0.1_0.9_1.csv'
mix3_results1 = evaluate_agent(filename,1,n_runs)
mix3_results2 = evaluate_agent(filename,2,n_runs)
mix3_results3 = evaluate_agent(filename,3,n_runs)

nextrow = ['0.1_0.9_1.0',np.mean(mix3_results1), np.mean(mix3_results2),np.mean(mix3_results3)]
resultsTable.loc[len(resultsTable)] = nextrow



#PRINTING RESULTS: 


print("Best for 1: ", one_results1)
print("Best for 2: ", two_results2)
print("Best for 3: ", three_results3)

'''

#PLOTTING THE BLOODY THING

df = pd.DataFrame()
df['One Step'] = one_results1/10
df['Two Step'] = two_results2/4
df['Three Step'] = three_results3/2
df.plot()
plt.title('Best Cases')
plt.show()




print("Trained on One, Test on Two: ", one_results2)
print("Trained on One, Test on Three: ", one_results3)
df = pd.DataFrame()
df['One Step'] = one_results1/10
df['Two Step'] = one_results2/4
df['Three Step'] = one_results3/2
df.plot()
plt.title('Trained on one')
plt.show()



print("Trained on Two, Test on One: ", two_results1)
print("Trained on Two, Test on Three: ", two_results3)

df = pd.DataFrame()
df['One Step'] = two_results1/10
df['Two Step'] = two_results2/4
df['Three Step'] = two_results3/2
df.plot()
plt.title('Trained on two')
plt.show()


print("Trained on Three, Test on One: ", three_results1)
print("Trained on Three, Test on Two: ", three_results2)
df = pd.DataFrame()
df['One Step'] = three_results1/10
df['Two Step'] = three_results2/4
df['Three Step'] = three_results3/2
df.plot()
plt.title('Trained on three')
plt.show()



print("10% 1, 90% 3, on 1 ", mix1_results1)
print("10%1, 90% 3, on 3", mix1_results3)


print("even mix on 1: ", mix2_results1)
print("even mix on 2: ", mix2_results2)
print("even_mix on 3: ", mix2_results3)


df['One Step'] = mix2_results1/10
df['Two Step'] = mix2_results2/4
df['Three Step'] = mix2_results3/2
df.plot()
plt.title('Trained on even mix of jumps')
plt.show()


df['One Step'] = mix1_results1/10
df['Three Step'] = mix1_results3/2
df.plot()
plt.title('Trained on 10% 1, 90% 3 of jumps')
plt.show()




df['One Step'] = mix3_results1/10
df['Two Step'] = mix3_results2/4
df['Three Step'] = mix3_results3/2
df.plot()
plt.title('Trained on 10% 1, 80% 2, 10% 3 of jumps')
plt.show()

'''


# BIG TABLE: 

print("du du du ... results: ")

print(resultsTable)


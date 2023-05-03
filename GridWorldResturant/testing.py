from resturantGridWorld import resturantGridWorld
import numpy as np 
import gym 
import matplotlib.pyplot as plt
import pandas as pd
import pdb 
from evaluation import evaluate_agent


folder = 'Q_Tables/Q_table'

#TEST ONE: Evaluating agent on models trained on only one agent: 


#HUMAN SPEED 1: 
filename = folder + '1.0_0.0_0.0.csv'
one_results1 = evaluate_agent(filename,1,100)
one_results2 = evaluate_agent(filename, 2, 100)
one_results3 = evaluate_agent(filename,3,100)

#HUMAN SPEED 2: 
filename = folder + '0.0_1.0_0.0.csv'
two_results1 = evaluate_agent(filename,1,100)
two_results2 = evaluate_agent(filename, 2, 100)
two_results3 = evaluate_agent(filename,3,100)


#HUMAN_SPEED 3: 
filename = folder + '0.0_0.0_1.0.csv'
three_results1 = evaluate_agent(filename,1,100)
three_results2 = evaluate_agent(filename, 2, 100)
three_results3 = evaluate_agent(filename,3,100)

#TEST 2: Evaluating models trained on a mix of human speeds with each human speed
#90% 3 and 10% 1 
#evaluate on each, plot the results in a graph: 

filename = folder + '0.1_0.0_1.0.csv'
mix1_results1 = evaluate_agent(filename,1,100)
mix1_results3 = evaluate_agent(filename,3,100)




#30% 1, 30% 2 30% 3 
#evaluate on each, plot the results in a graph:
filename = folder + '0.3_0.6_0.9.csv'
mix2_results1 = evaluate_agent(filename,1,100)
mix2_results2 = evaluate_agent(filename, 2, 100)
mix2_results3 = evaluate_agent(filename,3,100)

#10% 1, 80% 2, 10% 3 
#evaluate on each, plot the results in a graph: 

filename = folder + '0.1_0.9_0.1.csv'

#PRINTING RESULTS: 


print("Best for 1: ", one_results1)
print("Best for 2: ", two_results2)
print("Best for 3: ", three_results3)

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






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
one_results1 = evaluate_agent(filename,1,10)
one_results2 = evaluate_agent(filename, 2, 10)
one_results3 = evaluate_agent(filename,3,10)


one_results_all = [np.mean(one_results1), np.mean(one_results2), np.mean(one_results3)]
print(one_results_all)

data= {
    'Methods':["100% 1"],
    'jump 1': [np.mean(one_results1)],
    'jump 2': [np.mean(one_results2)],
    'jump 3': [np.mean(one_results3)]
    
    }



resultsTable = pd.DataFrame(data)
print(resultsTable)


print("AND THEN ")

filename = folder + '0.0_1.0_0.0.csv'
two_results1 = evaluate_agent(filename,1,10)
two_results2 = evaluate_agent(filename, 2, 10)
two_results3 = evaluate_agent(filename,3,10)

two_results_all = ["100% 2",np.mean(two_results1), np.mean(two_results2), np.mean(two_results3)]

resultsTable.loc[len(resultsTable)] = two_results_all

print(resultsTable)
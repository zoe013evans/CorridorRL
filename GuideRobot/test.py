import matplotlib.pyplot as plt
import pandas as pd
import pdb 
from evaluation import evaluate_agent


folder = 'Q_Tables/Q_table'
n_runs = 5

#TEST ONE: Evaluating agent on models trained on only one agent: 


print("1")
#HUMAN SPEED 1: 
filename = folder + '1.0_0.0_0.0.csv'
one_results2 = evaluate_agent(filename, 2, n_runs)

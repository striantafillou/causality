
# coding: utf-8

# In[22]:

import pickle
import pandas as pd

graph = True
figDir = 'figs'
treatment = 'quality'
outcome = 'mood';
confound = ['duration','mood_prev','stress_prev','energy_prev','focus_prev','activity_prev','daytype_prev']

# read data
with open('data.dat') as f:
    data = pickle.load(f)
f.close()

# removing nan rows
for i in range(len(data)):
    data[i] = data[i].dropna()
    data[i] = data[i].reset_index(drop=True)
    
# removing empty subjects
ind_empty = []
for i in range(len(data)):
    if data[i].shape[0]==0:
        ind_empty.append(i)
data = [i for j, i in enumerate(data) if j not in ind_empty]
print str(len(ind_empty))+' subjects removed due to lack of data'

matching = psm_causal_effects()


# In[20]:




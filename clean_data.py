
# coding: utf-8

# In[11]:

import pickle
import pandas as pd
import numpy as np

# read data
with open('data.dat') as f:
    data, subjects = pickle.load(f)
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
subjects = [subj for i, subj in enumerate(subjects) if i not in ind_empty]
print str(len(ind_empty))+' subjects removed due to lack of data'

with open('data_clean.dat','w') as f:
    pickle.dump([data, subjects], f)
f.close()


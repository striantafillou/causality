
# coding: utf-8

# In[7]:

import pickle
import pandas as pd
import numpy as np

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

# convert daytype from str to int
for i in range(len(data)):
    ind = np.where(data[i]['daytype']=='normal')[0]
    data[i].loc[ind,'daytype'] = 0
    ind = np.where(data[i]['daytype']=='partial')[0]
    data[i].loc[ind,'daytype'] = 1
    ind = np.where(data[i]['daytype']=='off')[0]
    data[i].loc[ind,'daytype'] = 2
    ind = np.where(data[i]['daytype_prev']=='normal')[0]
    data[i].loc[ind,'daytype_prev'] = 0
    ind = np.where(data[i]['daytype_prev']=='partial')[0]
    data[i].loc[ind,'daytype_prev'] = 1
    ind = np.where(data[i]['daytype_prev']=='off')[0]
    data[i].loc[ind,'daytype_prev'] = 2
    
with open('data_clean.dat','w') as f:
    pickle.dump(data, f)
f.close()


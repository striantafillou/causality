
# coding: utf-8

# In[10]:

import pickle
import pandas as pd
import numpy as np
import statsmodels.api as sm 
import statsmodels.formula.api as smf

# read data
with open('data_clean.dat') as f:
    data, subjects = pickle.load(f)
f.close()


# adding subject ids
for i in range(len(data)):
    #data[i]['subject'] = pd.Series(i+np.zeros(data[i].shape[0]), index=data[i].index, dtype=int)
    data[i]['subject'] = pd.Series(np.repeat(subjects[i],data[i].shape[0],axis=0), index=data[i].index, dtype=str)

del subjects

# concatenatig into a single dataframe
data_all = pd.concat(data, axis=0)
data_all = data_all.reset_index(drop=True)

# keeping only relevant variables
data_all = data_all[['subject','mood','quality','mood_prev', 'act_prev', 'stress_prev', 'energy_prev', 'daytype']]

# removing nan rows
data_all = data_all.dropna()
data_all = data_all.reset_index(drop=True)

n_boot = 10

#for i in range

md = smf.glm('mood ~ quality + act_prev + stress_prev + energy_prev + mood_prev + daytype', data_all)
mdf = md.fit()
print mdf.summary()


# In[9]:

data_all


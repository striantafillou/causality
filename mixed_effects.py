
# coding: utf-8

# In[82]:

import statsmodels.api as sm 
import statsmodels.formula.api as smf
import pickle
import pandas as pd
import numpy as np
from psm_causal_effects import psm_causal_effects

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

for i in range(len(data)):
    data[i]['subject'] = pd.Series(i+np.zeros(data[i].shape[0]), index=data[i].index, dtype=int)

# concatenatig into a single dataframe
data = pd.concat(data, axis=0)
data = data.reset_index(drop=True)

# remove extra columns
# data = data.loc[:,['subject','mood','quality']]
data['mood'] = data['mood'].astype(float)
data['mood_prev'] = data['mood_prev'].astype(float)
data['quality'] = data['quality'].astype(float)

md = smf.mixedlm('mood ~ quality', data, groups=data['subject'])
mdf = md.fit() 
print(mdf.summary())


# In[83]:

md = smf.mixedlm('quality ~ mood_prev', data, groups=data['subject'])
mdf = md.fit() 
print(mdf.summary())


# In[60]:

data = sm.datasets.get_rdataset('dietox', 'geepack').data
md = smf.mixedlm('Weight ~ Time', data, groups=data['Pig'])
mdf = md.fit() 
print(mdf.summary())


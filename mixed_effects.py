
# coding: utf-8

# In[101]:

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

# adding subject ids
for i in range(len(data)):
    data[i]['subject'] = pd.Series(i+np.zeros(data[i].shape[0]), index=data[i].index, dtype=int)

# concatenatig into a single dataframe
data = pd.concat(data, axis=0)
data = data.reset_index(drop=True)

# keeping only relevant variables
data = data[['subject','mood','quality','mood_prev']]

# removing nan rows
data = data.dropna()
data = data.reset_index(drop=True)

md = smf.mixedlm('mood ~ quality', data, groups=data['subject'])
mdf = md.fit() 
print(mdf.summary())


# In[102]:

md = smf.mixedlm('quality ~ mood_prev', data, groups=data['subject'])
mdf = md.fit() 
print(mdf.summary())


# In[103]:

data


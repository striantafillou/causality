
# coding: utf-8

# In[16]:

import statsmodels.api as sm
import statsmodels.formula.api as smf
import pickle
import pandas as pd
import numpy as np
from psm_causal_effects import psm_causal_effects

# read data
with open('data.dat') as f:
    data, subjects = pickle.load(f)
f.close()

dmax = 0
dmin = 1000
for i in range(len(data)):
    nq = np.sum(~np.isnan(data[i]['quality']))
    if nq>dmax:
        dmax = nq
    if nq<dmin and nq!=0:
        dmin = nq
        
data_all = pd.concat(data,axis=0)
data_all = data_all.reset_index(drop=True)

print dmin, dmax
print np.nanmean(data_all['mood']), np.nanstd(data_all['mood'])
print np.nanmean(data_all['quality']), np.nanstd(data_all['quality'])


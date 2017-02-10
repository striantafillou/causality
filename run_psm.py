
# coding: utf-8

# In[1]:

import pickle
import pandas as pd
import numpy as np
from psm_causal_effects import psm_causal_effects

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


# In[3]:

n_boot = 100

ind_set = range(len(data))

es_mean = np.array([])
for k in range(n_boot):
    
    print k,
    
    inds = np.random.choice(ind_set, size=len(data), replace=True)

    es = np.array([])
    for i in inds:

        treatment = np.array(data[i]['mood_prev'])
        outcome = np.array(data[i]['quality'])
        confound = np.array(pd.concat([data[i]['quality_prev'],data[i]['stress_prev']],axis=1))

        es = np.append(es,psm_causal_effects(treatment=treatment, outcome=outcome, confound=confound, scorefun='replacement'))

    es_mean = np.append(es_mean, np.mean(es))
    


# In[4]:

import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

print np.mean(es_mean)
plt.hist(es_mean,20);


# In[ ]:

es


# In[ ]:

from psm_causal_effects import psm_causal_effects

treatment, confound, model = psm_causal_effects(, , )


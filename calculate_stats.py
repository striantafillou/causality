
# coding: utf-8

# In[4]:

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

for i in range(len(data)):
    print data[i].size


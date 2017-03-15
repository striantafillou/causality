
# coding: utf-8

# In[24]:

import pickle
import pandas as pd
import numpy as np
from psm_causal_effects import psm_causal_effects
from extract_remove_nans import extract_remove_nans

# read data
with open('data.dat') as f:
    data, subjects = pickle.load(f)
f.close()

scorefunction = 'replacement'
output = 'linear'
var_threshold = 0.5

inds = range(len(data))
#inds = range(10)

es_s2m = np.zeros([len(data),4])
es_m2s = np.zeros([len(data),4])
es_s2m_um = np.zeros([len(data), 1])
es_m2s_um = np.zeros([len(data), 1])

for (c,i) in enumerate(inds):

    if np.nanstd(data[i]['mood'])<var_threshold or np.nanstd(data[i]['quality'])<var_threshold:
        print 'skipping subject '+str(i)+' due to low variance in mood or sleep quality.'
        es_m2s_um[c] = np.nan
        es_m2s[c,:] = np.nan
        es_s2m_um[c] = np.nan
        es_s2m[c,:] = np.nan
        continue

    # mood on sleep - lag 0 - unmatched
    treatment = 'mood_prev'
    outcome = 'quality'
    data_s = extract_remove_nans(data[i], [treatment]+[outcome])
    es_m2s_um[c] = psm_causal_effects(treatment=data_s[treatment], outcome=data_s[outcome], confound=[], scorefun='unmatched', output=output)

    # mood on sleep - lag 0
    confound = ['dow', 'mean_temp_prev2', 'clear_prev2', 'act_prev2','daytype','stress_prev2','energy_prev2','focus_prev2']
    data_s = extract_remove_nans(data[i], [treatment]+[outcome]+confound)
    es_m2s[c,0] = psm_causal_effects(treatment=data_s[treatment], outcome=data_s[outcome], confound=data_s[confound], scorefun=scorefunction, output=output)

    # mood on sleep - lag 1
    confound = ['dow', 'mean_temp_prev2', 'clear_prev2', 'act_prev2','daytype','stress_prev2','energy_prev2','focus_prev2','mood_prev2','quality_prev']
    data_s = extract_remove_nans(data[i], [treatment]+[outcome]+confound)
    es_m2s[c,1] = psm_causal_effects(treatment=data_s[treatment], outcome=data_s[outcome], confound=data_s[confound], scorefun=scorefunction, output=output)

    # mood on sleep - lag 2
    confound = ['dow', 'mean_temp_prev2', 'clear_prev2', 'act_prev2','daytype','stress_prev2','energy_prev2','focus_prev2','mood_prev2','quality_prev','mood_prev3','quality_prev2']
    data_s = extract_remove_nans(data[i], [treatment]+[outcome]+confound)
    es_m2s[c,2] = psm_causal_effects(treatment=data_s[treatment], outcome=data_s[outcome], confound=data_s[confound], scorefun=scorefunction, output=output)

    # mood on sleep - lag 3
    confound = ['dow', 'mean_temp_prev2', 'clear_prev2', 'act_prev2','daytype','stress_prev2','energy_prev2','focus_prev2','mood_prev2','quality_prev','mood_prev3','quality_prev2','mood_prev4','quality_prev3']
    data_s = extract_remove_nans(data[i], [treatment]+[outcome]+confound)
    es_m2s[c,3] = psm_causal_effects(treatment=data_s[treatment], outcome=data_s[outcome], confound=data_s[confound], scorefun=scorefunction, output=output)

    ################ sleep on mood

    # sleep on mood - lag 0 - unmatched
    treatment = 'quality'
    outcome = 'mood'
    data_s = extract_remove_nans(data[i], [treatment]+[outcome])
    es_s2m_um[c] = psm_causal_effects(treatment=data_s[treatment], outcome=data_s[outcome], confound=[], scorefun='unmatched', output=output)

    # sleep on mood - lag 0
    confound = ['dow', 'mean_temp_prev', 'clear_prev', 'act_prev','daytype','stress_prev','energy_prev','focus_prev']
    data_s = extract_remove_nans(data[i], [treatment]+[outcome]+confound)
    es_s2m[c,0] = psm_causal_effects(treatment=data_s[treatment], outcome=data_s[outcome], confound=data_s[confound], scorefun=scorefunction, output=output)

    # sleep on mood - lag 1
    confound = ['dow', 'mean_temp_prev', 'clear_prev','act_prev','daytype','stress_prev','energy_prev','focus_prev','mood_prev','quality_prev']
    data_s = extract_remove_nans(data[i], [treatment]+[outcome]+confound)
    es_s2m[c,1] = psm_causal_effects(treatment=data_s[treatment], outcome=data_s[outcome], confound=data_s[confound], scorefun=scorefunction, output=output)

    # sleep on mood - lag 2
    confound = ['dow', 'mean_temp_prev', 'clear_prev','act_prev','daytype','stress_prev','energy_prev','focus_prev','mood_prev','quality_prev','mood_prev2','quality_prev2']
    data_s = extract_remove_nans(data[i], [treatment]+[outcome]+confound)
    es_s2m[c,2] = psm_causal_effects(treatment=data_s[treatment], outcome=data_s[outcome], confound=data_s[confound], scorefun=scorefunction, output=output)

    # sleep on mood - lag 3
    confound = ['dow', 'mean_temp_prev', 'clear_prev','act_prev','daytype','stress_prev','energy_prev','focus_prev','mood_prev','quality_prev','mood_prev2','quality_prev2','mood_prev3','quality_prev3']
    data_s = extract_remove_nans(data[i], [treatment]+[outcome]+confound)
    es_s2m[c,3] = psm_causal_effects(treatment=data_s[treatment], outcome=data_s[outcome], confound=data_s[confound], scorefun=scorefunction, output=output)

    
    

    


# In[64]:

import operator
a = np.abs(es_s2m_um - es_s2m)

indices = np.argsort(-a, axis=0)
indices



# In[143]:

# good:69, 
i =29 


print es_s2m_um[i]
print es_s2m[i, ]
print data[i][treatment].size



# In[135]:

treatment = 'quality'
outcome = 'mood'
confound = ['dow', 'mean_temp_prev', 'clear_prev', 'act_prev','daytype','stress_prev','energy_prev','focus_prev']
data1 = extract_remove_nans(data[i], [treatment]+[outcome]+confound)
score1, ind_t1, ind_c1 = psm_causal_effects(treatment=data1[treatment], outcome=data1[outcome],                                 confound=data1[confound], scorefun=scorefunction, output=output, return_indices=True)
print ind_t1, ind_c1


# In[139]:

import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

np.random.seed(seed=1)

noise_amp_mood = 0.05#0.075
noise_amp_quality = 0.15#0.2


plt.figure(figsize=[8,8])

#psm
#score1, ind_t1, ind_c1 = psm_causal_effects(treatment=data1['quality'], outcome=data1['mood'], confound=data1[['mood_prev','stress_prev','daytype','dow','act_prev','quality_prev','mean_temp']], scorefun='replacement', output='difference', return_indices=True)

plt.xlabel('Sleep Quality')
plt.ylabel('Next-day Mood')
plt.xlim([0,8])
plt.ylim([0,8])

#plt.legend(['subject 1','subject 2','pooled','mixed'],bbox_to_anchor=(1.4, 1))

#noise vectors
n_quality = noise_amp_quality*np.random.randn(data1[treatment].size)
n_mood = noise_amp_mood*np.random.randn(data1[treatment].size)

plt.plot(data1['quality']+n_quality,data1['mood']+n_mood,'o',markersize=8,        markeredgecolor =(0,0,1),alpha=.75,markerfacecolor=(1,1,1))
plt.plot(data1['quality'][ind_t1]+n_quality[ind_t1],data1['mood'][ind_t1]+n_mood[ind_t1],'x',markersize=8,        markeredgecolor =(0,0,1),alpha=.75,markerfacecolor=(1,1,1))
plt.plot(data1['quality'][ind_c1]+n_quality[ind_c1],data1['mood'][ind_c1]+n_mood[ind_c1],'x',markersize=8,        markeredgecolor =(1,0,1),alpha=.75,markerfacecolor=(0,0,1))



plt.ylim([0,8])
plt.xlim([0,8])



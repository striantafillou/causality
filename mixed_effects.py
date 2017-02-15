
# coding: utf-8

# In[95]:

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

# adding subject ids
for i in range(len(data)):
    #data[i]['subject'] = pd.Series(i+np.zeros(data[i].shape[0]), index=data[i].index, dtype=int)
    data[i]['subject'] = pd.Series(np.repeat(subjects[i],data[i].shape[0],axis=0), index=data[i].index, dtype=str)

del subjects

# concatenatig into a single dataframe
data_all = pd.concat(data, axis=0)
data_all = data_all.reset_index(drop=True)

# keeping only relevant variables
data_all = data_all[['subject','mood','quality','mood_prev']]

# removing nan rows
data_all = data_all.dropna()
data_all = data_all.reset_index(drop=True)

# load assessments
with open('../CS120/Assessment/assessment.dat') as f:
    ass = pickle.load(f)
f.close()


# In[94]:

len(data)


# In[122]:

import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

md = smf.mixedlm('mood ~ quality', data_all, groups=data_all['subject'], re_formula="~quality")
mdf = md.fit() 
print mdf.summary()

RE = np.array(mdf.random_effects)

plt.hist(RE,10);



# In[127]:

subjects = np.unique(data_all['subject'])
plt.figure(figsize=[5,5])
ind = np.zeros(subjects.size)
for i, subj in enumerate(subjects):
    ind[i] = np.where(ass['ID']==subj)[0]
    plt.plot(RE[i,1], np.nanmean(ass.loc[ind[i],'PHQ9 W0']), '.', markersize=2, color=(0,0,0))
plt.figure(figsize=[5,5])
for i, subj in enumerate(subjects):
    plt.plot(RE[i,1], np.nanmean(ass.loc[ind[i],'GAD7 W0']), '.', markersize=2, color=(0,0,0))


# In[128]:

md = smf.mixedlm('quality ~ mood_prev', data_all, groups=data_all['subject'], re_formula="~mood_prev")
mdf = md.fit() 
print mdf.summary()
RE = np.array(mdf.random_effects)
# plt.hist(RE,10);


# In[133]:

subjects = np.unique(data_all['subject'])
plt.figure(figsize=[5,5])
ind = np.zeros(subjects.size)
phq = np.zeros(subjects.size)
for i, subj in enumerate(subjects):
    ind[i] = np.where(ass['ID']==subj)[0]
    phq[i] = np.nanmean(ass.loc[ind[i],'PHQ9 W0'])
plt.plot(RE[:,1], phq, '.', markersize=2, color=(0,0,0))
print np.corrcoef(RE[:,1], phq)
plt.figure(figsize=[5,5])
for i, subj in enumerate(subjects):
    plt.plot(RE[i,1], np.nanmean(ass.loc[ind[i],'GAD7 W0']), '.', markersize=2, color=(0,0,0))


# In[9]:

betas_m2s = np.zeros([len(data),2])
betas_s2m = np.zeros([len(data),2])
for iSubj in range(len(data)):
    if data[iSubj].shape[0] ==0:
        betas_m2s[iSubj, :] = np.array([np.nan, np.nan])
        betas_s2m[iSubj, :] = np.array([np.nan, np.nan])
    else:
        md = smf.glm('quality ~ mood_prev', data[iSubj])
        mdf = md.fit()
        betas_m2s[iSubj, :] = mdf.params
        md = smf.glm('mood ~ quality', data[iSubj])
        mdf = md.fit()
        betas_s2m[iSubj, :] = mdf.params
    


# In[25]:

md = smf.glm('quality ~ mood_prev', data_all)
mdf = md.fit()
print mdf.summary()

plt.figure(figsize=[5,5])
for iSubj in range(len(data)):
    plt.plot([0,8], [betas_m2s[iSubj, 0],betas_m2s[iSubj,0]+betas_m2s[iSubj,1]*8], linewidth=.3, alpha=.5)
plt.plot([0,8], [mdf.params[0],mdf.params[0]+mdf.params[1]*8], color=(.2, .2,.2), linewidth=3)
plt.plot(data_all['mood_prev']+0.075*np.random.randn(data_all.shape[0]),data_all['quality']+0.2*np.random.randn(data_all.shape[0]),'.',markersize=2,        color=(0,0,0),alpha=.5)
plt.xlabel('Mood')
plt.ylabel('Sleep Quality')
plt.xlim([0,8])
plt.ylim([0,8])


# In[54]:

md = smf.glm('mood ~ quality', data_all)
mdf = md.fit()
print mdf.summary()

plt.figure(figsize=[5,5])
for iSubj in range(len(data)):
    plt.plot([0,8], [betas_s2m[iSubj, 0],betas_s2m[iSubj,0]+betas_s2m[iSubj,1]*8], linewidth=.3, alpha=.5)
plt.plot([0,8], [mdf.params[0],mdf.params[0]+mdf.params[1]*8], color=(.2,.2,.2), linewidth=3)
plt.plot(data_all['quality']+0.2*np.random.randn(data_all.shape[0]),data_all['mood']+0.075*np.random.randn(data_all.shape[0]),'.',markersize=2,        color=(0,0,0),alpha=.5)
plt.xlabel('Sleep Quality')
plt.ylabel('Mood')
plt.xlim([0,8])
plt.ylim([0,8])



# In[121]:

ind1 = 49
ind2 = 149

plt.figure(figsize=[5,5])
plt.plot([0,8], [betas_s2m[ind1, 0],betas_s2m[ind1,0]+betas_s2m[ind1,1]*8], linewidth=1, alpha=.5, color=(0,0,1))
plt.plot([0,8], [betas_s2m[ind2, 0],betas_s2m[ind2,0]+betas_s2m[ind2,1]*8], linewidth=1, alpha=.5, color=(1,0,0))
#plt.plot([0,8], [mdf.params[0],mdf.params[0]+mdf.params[1]*8], color=(.2,.2,.2), linewidth=1)

data_2  = pd.concat([data[ind1], data[ind2]],axis=0)
# keeping only relevant variables
data_2 = data_2[['subject','mood','quality','mood_prev']]
data_2 = data_2.dropna()
data_2.reset_index(drop=True)

#pooled regression
md = smf.glm('mood ~ quality', data_2)
mdf = md.fit()
plt.plot([0,8], [mdf.params[0],mdf.params[0]+mdf.params[1]*8], color=(.2,.2,.2), linewidth=1, linestyle='--')

#mixed linear models regression
md = smf.mixedlm('quality ~ mood_prev', data_2, groups=data_2['subject'], re_formula="~mood_prev")
mdf = md.fit() 
plt.plot([0,8], [mdf.params[0],mdf.params[0]+mdf.params[1]*8], color=(.2,.2,.2), linewidth=1)

plt.xlabel('Sleep Quality')
plt.ylabel('Mood')
plt.xlim([0,8])
plt.ylim([0,8])

plt.legend(['subject 1','subject 2','pooled','mixed'],bbox_to_anchor=(1.4, 1))

plt.plot(data[ind1]['quality']+0.2*np.random.randn(data[ind1].shape[0]),data[ind1]['mood']+0.075*np.random.randn(data[ind1].shape[0]),'.',markersize=5,        color=(0,0,1),alpha=.5)
plt.plot(data[ind2]['quality']+0.2*np.random.randn(data[ind2].shape[0]),data[ind2]['mood']+0.075*np.random.randn(data[ind2].shape[0]),'.',markersize=5,        color=(1,0,0),alpha=.5)


# In[63]:

mdf.params


# In[ ]:




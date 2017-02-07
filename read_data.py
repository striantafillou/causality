
# coding: utf-8

# In[ ]:

import os
import pickle
import pandas as pd
import numpy as np
import datetime, time

data_dir = '/data/CS120/'
subjects = os.listdir(data_dir)
# subjects.remove('.dropbox')
# subjects = subjects[:2]

timezones = pd.read_csv('../CS120/general/timezones.csv',sep='\t',header=None)
ind_nan = np.where(np.isnan(timezones[1]))[0]
timezones.loc[ind_nan,1]=0

acts = []
emas = []
emss = []

for subj in subjects:
    
    print subj
    
    act = pd.DataFrame(columns=['date','act'])
    if os.path.exists(data_dir+subj+'/act.csv'):
        data = pd.read_csv(data_dir+subj+'/act.csv',sep='\t',header=None)
        # convert timestamps to daystamps
        data[0] = np.floor((data[0]+ 3600*float(timezones.loc[timezones[0]==subj,1]))/86400.0)
        # loading into new matrix
        act['date'] = np.arange(data.loc[0,0],data.loc[data.shape[0]-1,0])
        for (i,da) in enumerate(act['date']):
             act.loc[i,'act'] = np.sum(data.loc[data[0]==da,1]=='BIKING')+np.sum(data.loc[data[0]==da,1]=='ON_FOOT')
    else:
        print ' no act data'
    acts.append(act)


    ema = pd.DataFrame(columns=['date','stress','mood','energy','focus'])
    if os.path.exists(data_dir+subj+'/emm.csv'):
        data = pd.read_csv(data_dir+subj+'/emm.csv',sep='\t',header=None)
        # convert timestamps to daystamps
        data[0] = np.floor((data[0]+ 3600*float(timezones.loc[timezones[0]==subj,1]))/86400.0)
        # loading into new matrix
        ema['date'] = np.arange(data.loc[0,0],data.loc[data.shape[0]-1,0])
        for (i,da) in enumerate(ema['date']):
            ema.loc[i,'stress'] = np.nanmean(data.loc[data[0]==da,1])
            ema.loc[i,'mood'] = np.nanmean(data.loc[data[0]==da,2])
            ema.loc[i,'energy'] = np.nanmean(data.loc[data[0]==da,3])
            ema.loc[i,'focus'] = np.nanmean(data.loc[data[0]==da,4])
    else:
        print ' no ema data'
    emas.append(ema)
        
    ems = pd.DataFrame(columns=['date','duration','quality','daytype'])
    if os.path.exists(data_dir+subj+'/ems.csv'):
        data = pd.read_csv(data_dir+subj+'/ems.csv',sep='\t',header=None)
        # convert timestamps to daystamps
        data[0] = np.floor((data[3]/1000.0+3600*float(timezones.loc[timezones[0]==subj,1]))/86400.0)
        # loading into new matrix
        ems['date'] = np.arange(data.loc[0,0],data.loc[data.shape[0]-1,0])
        for (i,da) in enumerate(ems['date']):
            ems.loc[i,'duration'] = np.nanmean(data.loc[data[0]==da,3]-data.loc[data[0]==da,2])/1000.0
            ems.loc[i,'quality'] = np.nanmean(data.loc[data[0]==da,5])
            if data.loc[data[0]==da,6].size>0:
                ems.loc[i,'daytype'] = data.loc[data[0]==da,6].values[0]
            else:
                ems.loc[i,'daytype'] = np.nan
    else:
        print ' no ems data'
    emss.append(ems)

# aligning the data
data = []
for (i,_) in enumerate(subjects):
    a = pd.merge(emas[i],emss[i],on='date',how='outer')
    a = pd.merge(a,acts[i],on='date',how='outer')
    
    # delayed version
    emas[i]['date'] += 1
    emss[i]['date'] += 1
    acts[i]['date'] += 1
    emas[i].columns = ['date','stress_prev','mood_prev','energy_prev','focus_prev']
    emss[i].columns = ['date','duration_prev','quality_prev','daytype_prev']
    acts[i].columns = ['date','act_prev']
    
    a = pd.merge(a,emas[i],on='date',how='outer')
    a = pd.merge(a,emss[i],on='date',how='outer')
    a = pd.merge(a,acts[i],on='date',how='outer')
    
    data.append(a)
    
with open('data.dat','w') as f:
    pickle.dump(data, f)
f.close()


# In[ ]:

i


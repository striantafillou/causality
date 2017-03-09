
# coding: utf-8

# In[1]:

def daytype2number(x):
    return {
        'normal': 0,
        'partial': 1,
        'off': 2,
    }[x]


# In[2]:

import os
import pickle
import pandas as pd
import numpy as np
import datetime, time

data_dir ='/data/CS120/'
weather_dir ='/data/CS120Weather/'
csv_file = '../CS120/general/timezones.csv'
# data_dir = '../../../Data/depression2016/CS120Data/CS120/'
# csv_file = '../../../Data/depression2016/CS120Data/timezones.csv'
# weather_dir ='../../../Data/depression2016/CS120Weather/'

subjects = os.listdir(data_dir)
timezones = pd.read_csv(csv_file,sep='\t',header=None)


# In[3]:

# subjects = subjects[:2]

ind_nan = np.where(np.isnan(timezones[1]))[0]
timezones.loc[ind_nan,1]=0

acts = []
emas = []
emss = []
wtrs =[]

for subj in subjects:
    
    print subj,
    
    act = pd.DataFrame(columns=['date','act'], dtype=float)
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

    ema = pd.DataFrame(columns=['date','stress','mood','energy','focus'], dtype=float)
    if os.path.exists(data_dir+subj+'/emm.csv'):
        data = pd.read_csv(data_dir+subj+'/emm.csv',sep='\t',header=None)
        # convert timestamps to daystamps
        data[0] = np.floor((data[0]+ 3600*float(timezones.loc[timezones[0]==subj,1]))/86400.0)
        # loading into new matrix
        ema['date'] = np.arange(data.loc[0,0],data.loc[data.shape[0]-1,0])
        for (i,da) in enumerate(ema['date']):
            ema.loc[i,'stress'] = np.nanmean(data.loc[data[0]==da,1])
            moods = data.loc[data[0]==da,2]
            ema.loc[i,'mood'] = np.nanmean(moods)
            ema.loc[i,'energy'] = np.nanmean(data.loc[data[0]==da,3])
            ema.loc[i,'focus'] = np.nanmean(data.loc[data[0]==da,4])
    else:
        print ' no ema data'
    emas.append(ema)
    
    ems = pd.DataFrame(columns=['date','duration','quality','daytype'], dtype=float)
    if os.path.exists(data_dir+subj+'/ems.csv'):
        data = pd.read_csv(data_dir+subj+'/ems.csv',sep='\t',header=None)
        # convert timestamps to daystamps
        data[0] = np.floor((data[3]/1000.0+3600*float(timezones.loc[timezones[0]==subj,1]))/86400.0)
        # loading into new matrix
        ems['date'] = np.arange(data.loc[0,0],data.loc[data.shape[0]-1,0])
        for (i,da) in enumerate(ems['date']):
            ems.loc[i,'duration'] = np.nanmean(data.loc[data[0]==da,3]-data.loc[data[0]==da,2])/1000.0
            qual = np.array(data.loc[data[0]==da,5])
            # if multiple entries, only take the first one
            if qual.size>1:
                ems.loc[i,'quality'] = qual[0]
            else:
                ems.loc[i,'quality'] = np.nanmean(qual)
            daytype = data.loc[data[0]==da,6]
            if daytype.size>0:
                ems.loc[i,'daytype'] = daytype2number(daytype.values[0])
            else:
                ems.loc[i,'daytype'] = np.nan
    else:
        print ' no ems data'
    emss.append(ems)
        
    wtr = pd.DataFrame(columns=['date','mean_temp','clear'], dtype=float)
    if os.path.exists(weather_dir+subj+'/wtr.csv') and  os.stat(weather_dir+subj+'/wtr.csv').st_size > 0:
        data = pd.read_csv(weather_dir+subj+'/wtr.csv',sep='\t',header=None)
        # convert timestamps to daystamps
        data[0] = np.floor((data[0]+3600*float(timezones.loc[timezones[0]==subj,1]))/86400.0)
        # loading into new matrix
        wtr['date'] = np.arange(data.loc[0,0],data.loc[data.shape[0]-1,0])
        for (i,da) in enumerate(wtr['date']):
            # wrt.loc[i,'duration'] = np.nanmean(data.loc[data[0]==da,1])/1000.0
            tmptemp = np.array(data.loc[data[0]==da,1])
            tmpclear = np.array(data.loc[data[0]==da, 9])

            # take the mean of multiple entries
            if tmptemp.size ==0:
#                 print i, 'no temperature data'
            else:
                wtr.loc[i,'mean_temp'] = np.nanmean(tmptemp)
                wtr.loc[i, 'clear'] = np.sum(tmpclear=='Clear');
    else:
        print ' no weather data'
    wtrs.append(wtr)


# In[4]:

# aligning the data

data = []
for (i,_) in enumerate(subjects):
    
    a = pd.merge(emas[i],emss[i],on='date',how='outer')
    a = pd.merge(a,acts[i],on='date',how='outer')
    a = pd.merge(a,wtrs[i], on='date', how='outer')
    
    # delayed (-1)
    emas[i]['date'] += 1
    emss[i]['date'] += 1
    acts[i]['date'] += 1
    wtrs[i]['date'] += 1
    emas[i].columns = ['date','stress_prev','mood_prev','energy_prev','focus_prev']
    emss[i].columns = ['date','duration_prev','quality_prev','daytype_prev']
    acts[i].columns = ['date','act_prev']
    wtrs[i].columns = ['date','mean_temp_prev', 'clear_prev']
    a = pd.merge(a,emas[i],on='date',how='outer')
    a = pd.merge(a,emss[i],on='date',how='outer')
    a = pd.merge(a,acts[i],on='date',how='outer')
    a = pd.merge(a,wtrs[i], on='date', how='outer')

    # removing extra columns
    emss[i] = emss[i].drop(['duration_prev','daytype_prev'], axis=1)
    
    # delayed (-2)
    emas[i]['date'] += 1
    emss[i]['date'] += 1
    acts[i]['date'] += 1
    wtrs[i]['date'] += 1
    emas[i].columns = ['date','mood_prev2','stress_prev2','energy_prev2','focus_prev2']
    emss[i].columns = ['date','quality_prev2']
    acts[i].columns = ['date','act_prev2']
    wtrs[i].columns = ['date','mean_temp_prev2', 'clear_prev2']
    a = pd.merge(a,emas[i],on='date',how='outer')
    a = pd.merge(a,emss[i],on='date',how='outer')
    a = pd.merge(a,acts[i],on='date',how='outer')
    a = pd.merge(a,wtrs[i],on='date',how='outer')

    # removing extra columns
    emas[i] = emas[i].drop(['stress_prev2','energy_prev2','focus_prev2'], axis=1)
    
    # delayed (-3)
    emas[i]['date'] += 1
    emss[i]['date'] += 1
    emas[i].columns = ['date','mood_prev3']
    emss[i].columns = ['date','quality_prev3']
    a = pd.merge(a,emas[i],on='date',how='outer')
    a = pd.merge(a,emss[i],on='date',how='outer')

    # delayed (-4) - for mood only
    emas[i]['date'] += 1
    emas[i].columns = ['date','mood_prev4']
    a = pd.merge(a,emas[i],on='date',how='outer')
    
    data.append(a)


# In[5]:

# add day of the week and write to disk

import datetime as dt
import calendar

for (i,_) in enumerate(subjects):
    ts =data[i].date*86400
    #tmp =dt.datetime.fromtimestamp().day
    data[i]['dow']=[dt.datetime.fromtimestamp(t).weekday() for t in ts]
    
with open('data.dat','w') as f:
    pickle.dump([data, subjects], f)
f.close()


# In[6]:

data[0]


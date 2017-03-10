
# coding: utf-8

# In[35]:

import numpy as np
import pandas as pd
import xgboost as xgb
import pickle

n_fold = 10
xvars = ['quality','dow', 'act_prev','daytype','stress_prev','energy_prev','focus_prev','mood_prev','quality_prev','mood_prev2','quality_prev2','mood_prev3','quality_prev3']
yvars = ['mood']
# read data
with open('data.dat') as f:
    data, subjects = pickle.load(f)
f.close()

# concatenatig into a single dataframe
data_all = pd.concat(data, axis=0)
data_all = data_all.reset_index(drop=True)

# data = data[:10]

# personal analysis
for i in range(len(data)):
    
    
    data[i] = data[i][xvars+yvars]
    data[i] = data[i].dropna()
    data[i] = data[i].reset_index(drop=True)
    
    if data[i].shape[0]<20:
        print 'skipping subjetc '+subjects[i]
        continue
    
    x = data[i][xvars]
    y = data[i][yvars]

    
    
    fold_size = np.floor(data[i].shape[0]/float(n_fold))
#     print fold_size
    
    r2 = np.zeros(n_fold)
    
    for j in range(n_fold):
        
        xtest = x.loc[j*fold_size:(j+1)*fold_size-1,:]
        xtest = xtest.reset_index(drop=True)
        ytest = y.loc[j*fold_size:(j+1)*fold_size-1]
        ytest = ytest.reset_index(drop=True)
        
        xtrain = x.loc[0:j*fold_size-1,:]
        xtrain = pd.concat([xtrain, x.loc[(j+1)*fold_size:,:]], axis=0)
        xtrain = xtrain.reset_index(drop=True)

        ytrain = y.loc[0:j*fold_size-1]
        ytrain = pd.concat([ytrain, y.loc[(j+1)*fold_size:]], axis=0)
        ytrain = ytrain.reset_index(drop=True)
        
        gbm = xgb.XGBRegressor(max_depth=3, learning_rate=0.1, n_estimators=50, silent=True, objective='reg:linear', nthread=-1,                         gamma=0, min_child_weight=1, max_delta_step=0, subsample=0.25, colsample_bytree=1, colsample_bylevel=1,                         reg_alpha=0, reg_lambda=1, scale_pos_weight=1, base_score=0.5, seed=0, missing=None)
        gbm.fit(xtrain, ytrain, eval_set=[(xtrain,ytrain),(xtest, ytest)], eval_metric='rmse', verbose=False)
        ypred = gbm.predict(xtest)
        
        r2[j] = 1 - np.sum(np.power(ypred-np.array(ytest),2))/np.sum(np.power(np.mean(ytrain)-ytest,2))
        
    print i, np.mean(r2)
        
# mixed analysis


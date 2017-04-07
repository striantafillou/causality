
# coding: utf-8

# In[4]:

import numpy as np
from sklearn import linear_model
import statsmodels.formula.api as smf
import pandas as pd

def psm_causal_effects(treatment, outcome, confound, graph=0, scorefun='replacement', caliper=0.1, output='difference', return_indices=False):
    
    if treatment.size<20:
#         print 'too few samples!'
        return np.nan

    if np.var(treatment)==0:
        print('treatment has no variability')
        return np.nan

    #binarizing the treatment
    treatment_bin = 0.0*(treatment < np.mean(treatment)) + 1.0*(treatment >= np.mean(treatment))
    treatment_bin = treatment_bin.astype(float)

    ind_case = np.where(treatment_bin==1)[0]
    ind_control = np.where(treatment_bin==0)[0]
    
#     print ind_case
#     print ind_control

    if scorefun=='unmatched':
        
        ind_matched_case = ind_case
        ind_matched_control = ind_control
    
    else:
    
        #fitting on confounds
        model = linear_model.LogisticRegression()

        model.fit(confound, treatment_bin)

        #prediction
        pscore = model.predict_proba(confound)[:,1]
        
        ind_matched_case = np.array([],dtype=int)
        ind_matched_control = np.array([],dtype=int)
        
        for i_case in ind_case:

            #finding closest match in control
            diffs = abs(pscore[i_case]-pscore[ind_control])
            ind = np.argmin(diffs)

            #check if matches are close enough when no replacement
            if scorefun=='noreplacement' and diffs[ind]>=caliper:
                continue

            ind_matched_control = np.append(ind_matched_control,ind_control[ind])
            ind_matched_case = np.append(ind_matched_case,i_case)
            
            # if no-replacement remove the used sample
            if scorefun=='noreplacement':
                temp = list(ind_control)
#                 if len(temp)>1:
#                     break
                temp.remove(ind_control[ind])
                ind_control = np.array(temp, dtype=int)
                if ind_control.size==0:
                    break

    if output=='difference':
        # estimate psm effect size
        std_pooled = np.var(outcome[ind_matched_case])*(ind_matched_case.size-1) + np.var(outcome[ind_matched_control])*(ind_matched_control.size-1)
        std_pooled /= (ind_matched_case.size+ind_matched_control.size-2)
        std_pooled = np.sqrt(std_pooled)
        out = (np.mean(outcome[ind_matched_case])-np.mean(outcome[ind_matched_control]))/std_pooled
    elif output=='linear':
        # estimate regression coefficients
#         print ind_matched_control.shape, ind_matched_case.shape
        treatment = treatment.loc[np.concatenate((ind_matched_control, ind_matched_case))]
        outcome = outcome.loc[np.concatenate((ind_matched_control, ind_matched_case))]
#         regr = linear_model.LinearRegression()
#         regr.fit(np.array(treatment).reshape(treatment.size,1), np.array(outcome).reshape(outcome.size,1))
#         out = regr.coef_
        data = pd.DataFrame({'treatment': np.array(treatment), 'outcome': np.array(outcome)})
        md = smf.glm('outcome ~ treatment', data)
        mdf = md.fit()
        out = mdf.params[1]
    else:
        print('warning: unknown output type')
        out = np.nan
    
    if return_indices==True:
        return out, ind_matched_case, ind_matched_control
    else:
        return out


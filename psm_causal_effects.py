
# coding: utf-8

# In[4]:

import numpy as np
from sklearn import linear_model

def psm_causal_effects(treatment, outcome, confound, graph=0, scorefun='replacement', caliper=0.2):
    
    if treatment.size<20:
#         print 'too few samples!'
        return np.nan

    if np.var(treatment)==0:
        print 'treatment has no variability'
        return np.nan

    #binarizing the treatment
    treatment = 0*(treatment < np.mean(treatment)) + 1*(treatment >= np.mean(treatment))

    ind_case = np.where(treatment==1)[0]
    ind_control = np.where(treatment==0)[0]

    cnt_control = np.zeros(ind_control.size)
    
    if scorefun=='unmatched':
        
        ind_matched_case = ind_case
        ind_matched_control = ind_control
        cnt_control = np.ones(ind_control.size)
    
    else:
    
        #fitting on confounds
        model = linear_model.LogisticRegression()

        model.fit(confound, treatment)

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
            
            # counter for how many times a control is used
            cnt_control[ind] += 1

            # if without replacement remove the used sample
            if scorefun=='noreplacement':
                temp = list(ind_control)
                if len(temp)>1:
                    break
                temp.remove(ind_control[ind])
                ind_control = np.array(temp, dtype=int)

    #std_pooled = np.var(outcome[ind_matched_case])*(ind_matched_case.size-1) + np.var(outcome[ind_matched_control])*(ind_matched_control.size-1)
    #std_pooled /= (ind_matched_case.size+ind_matched_control.size-2)
    #std_pooled = np.sqrt(std_pooled)
    
    # accounting for multiplicity in controls
    ind_matched_control_uniq = np.unique(ind_matched_control)
    std_pooled = np.var(outcome[ind_matched_case])/float(ind_matched_case.size) +        np.var(outcome[ind_matched_control_uniq])*np.sum(np.multiply(cnt_control,cnt_control))/float(np.sum(cnt_control)**2)
    std_pooled = np.sqrt(std_pooled)
    
    return (np.mean(outcome[ind_matched_case])-np.mean(outcome[ind_matched_control]))/std_pooled


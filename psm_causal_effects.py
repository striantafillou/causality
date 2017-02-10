
# coding: utf-8

# In[ ]:

import numpy as np
from sklearn import linear_model

def psm_causal_effects(treatment, outcome, confound, graph=0, scorefun='replacement', caliper=0.2):
    
    if treatment.size<20:
#         print 'too few samples!'
        return []

    #binarizing the treatment
    treatment = 0*(treatment < np.mean(treatment)) + 1*(treatment >= np.mean(treatment))

    #fitting on confounds
    model = linear_model.LogisticRegression()

    model.fit(confound, treatment)

    #prediction
    pscore = model.predict_proba(confound)[:,1]

    ind_case = np.where(treatment==1)[0]
    ind_control = np.where(treatment==0)[0]

    if scorefun=='unmatched':
        
        ind_matched_case = ind_case
        ind_matched_control = ind_control
    
    else:
    
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

            # if without replacement remove the used sample
            if scorefun=='noreplacement':
                temp = list(ind_control)
                if len(temp)>1:
                    break
                temp.remove(ind_control[ind])
                ind_control = np.array(temp, dtype=int)

    return np.mean(outcome[ind_matched_case])-np.mean(outcome[ind_matched_control])


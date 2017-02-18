
# coding: utf-8

# In[ ]:

import pandas as pd

def extract_remove_nans(data, cols):
    
    data = data[cols]
    
    data = data.dropna()
    data = data.reset_index(drop=True)
    
    return data
    


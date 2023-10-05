#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


# Run a conservative cloud detection
def qa_pixel_interp_aggressive(number):
    '''
    Helps interpret the 16bit data in the landsat qa pixels
    
    returns True if there is mid confidence cirrus, snow/ice, cloud shadow, OR clouds
    '''
    binary = bin(number)[2:].zfill(16)
    
    # if medium to high confidence cirrus, snow/ice, cloud shadow, and clouds
    if int(binary[:2]) > 1:
        return True
    elif int(binary[2:4]) > 1:
        return True
    elif int(binary[4:6]) > 1:
        return True
    elif int(binary[6:8]) > 1:
        return True
    else:
        return False
    
def qa_pixel_interp_conservative(number):
    '''
    Helps interpret the 16bit data in the landsat qa pixels
    
    returns True if there is mid confidence cirrus, snow/ice, cloud shadow, OR clouds
    '''
    binary = bin(number)[2:].zfill(16)
    
    # if high confidence cirrus, snow/ice, cloud shadow, and clouds
    # 01 - low, 10 - medium, 11 - high
    if int(binary[:2]) > 10:
        return True
    elif int(binary[2:4]) > 10:
        return True
    elif int(binary[4:6]) > 10:
        return True
    elif int(binary[6:8]) > 10:
        return True
    else:
        return False
    
def qa_pixel_interp_conserv_water(number):
    '''
    Helps interpret the 16bit data in the landsat qa pixels
    
    returns True if there is mid confidence cirrus, snow/ice, cloud shadow, clouds OR WATER
    '''
    binary = bin(number)[2:].zfill(16)
    
    # if high confidence cirrus, snow/ice, cloud shadow, and clouds
    # 01 - low, 10 - medium, 11 - high
    if int(binary[:2]) > 10:
        return True
    elif int(binary[2:4]) > 10:
        return True
    elif int(binary[4:6]) > 10:
        return True
    elif int(binary[6:8]) > 10:
        return True
    # if water return true
    elif int(binary[8]) == 1:
        return True
    else:
        return False
    
def apply_array_func(func, x):
    '''
    Applies a function element-wise across a 1D array
    '''
    return np.array([func(xi) for xi in x])

def run_qa_parser(qa_raster, func):
    '''
    Accepts any array consisting of 16 bit unsigned integers.
    Generates a binary cloud mask using the function provided.
    Returns a squeezed binary cloud mask
    '''
    unique_vals = np.unique(qa_raster)
    masked_vals = apply_array_func(func, unique_vals)
    masked_vals = unique_vals[masked_vals]
    cl_mask = np.isin(qa_raster, masked_vals)
    
    return cl_mask.squeeze()


# In[ ]:





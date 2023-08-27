#!/usr/bin/env python
# coding: utf-8

# In[43]:


import rasterio as rio
from rasterio.plot import show
import os, fnmatch
import re
from tqdm import tqdm
import numpy as np


# In[45]:


### FUNCTIONS NEEDED ###

def parse_mtl(mtl_file):
    """
    Parses the landsat metadata file into a dictionary of dictionaries.
    
    Dictionary is split into several sub-dicts including PRODUCT_CONTENTS and IMAGE_ATTRIBUTES
    """
    
    with open(f"{mtl_file}.txt") as f:
            lines = f.readlines()
            f.close()

    clean_lines = [element.strip("\n").strip() for element in lines]

    ### PARSE THE MTL FILE INTO A DICTIONARY ###
    # Find all major groups in the metadata
    groups = [element for element in clean_lines if element.startswith("GROUP")]

    group_dict = dict()

    # We don't need the overarching metadata group
    for group in groups[1:]:
        # Return the part of list that the group contains
        contents = clean_lines[clean_lines.index(group)+1:clean_lines.index(f"END_{group}")]

        data_dict = {}
        # Iterate through the elements in the list
        for element in contents:
            # Split the element by "="
            parts = element.split("=")
            if len(parts) == 2:
                # Assign A as key and B as value to the dictionary
                key = parts[0].strip()  # Remove leading/trailing whitespace
                value = parts[1].strip()  # Remove leading/trailing whitespace
                data_dict[key] = value.strip("\"") # Remove quotation marks

        group_dict[group.replace("GROUP = ", "", 1)] = data_dict
    
    return group_dict

def toa_reflectance(raster, band_num, metadata):
    """
    raster: requires a 2D numpy array as read from rasterio
    NB - array should be masked since landsat uses 0 for np.nan
    
    band_num: the landsat band number associated with that raster
    
    returns the landsat level 1 product raster corrected for TOA
    Note that these are center image sun corrected - you can do pixel level sun correction but it
    takes a lot more work
    """
    # Get TOA reflectance
    toa_ref_no_suncorr = raster * float(metadata["LEVEL1_RADIOMETRIC_RESCALING"][f"REFLECTANCE_MULT_BAND_{band_num}"]) + float(metadata["LEVEL1_RADIOMETRIC_RESCALING"][f"REFLECTANCE_ADD_BAND_{band_num}"])
    
    # Correct for sun elevation
    toa_ref = toa_ref_no_suncorr / np.sin(float(metadata["IMAGE_ATTRIBUTES"]["SUN_ELEVATION"]))
    
    # Clip any values that are larger than 1 to 1
    toa_ref[toa_ref > 1] = 1
    
    return toa_ref

def histogram_stretch(img, min_vals = None, max_vals = 99):
    """
    Performs a histogram_stretch on an image. DO NOT use this for analytical workflows - 
    this should only be used to improve image visualization
    
    img: an unmasked 3D raster 
    
    min_vals: percentile that you wish to crop to
        will be np.zeros by default
    max_vals: percentile that you wish to crop to
        will be np.nanpercentile(img, 99) by default # crops to 99th percentile
    """
    if img.ndim != 3:
        print("ValueError: Your raster must have three dimensions.")
        return
    
    # This returns the max_valth percentile
    max_vals = np.nanpercentile(img, max_vals, axis = (1,2)).reshape(img.shape[0],1,1) 
    # min_vals = np.nanmin(tcc_toa, axis = (1,2)).reshape(3,1,1) # Use this to stretch to minimum values
    if min_vals is not None:
        min_vals = np.nanpercentile(img, min_vals, axis = (1,2)).reshape(img.shape[0],1,1)
    else:
        min_vals = np.zeros(img.shape[0]).reshape(img.shape[0],1,1)
    
    # Perform normalization
    img_stretched = (img - min_vals) / (max_vals - min_vals)
    
    # Clip values above 1
    img_stretched[img_stretched > 1] = 1
    
    return img_stretched


# In[46]:



# filenames


# In[49]:


if __name__ == "__main__":
    ### RETRIEVE ALL LANDSAT FILENAMES IN THIS METADATA ###
    metadata = parse_mtl("LC08_L1TP_113037_20230502_20230509_02_T1_MTL")

    filenames = [value for key, value in metadata["PRODUCT_CONTENTS"].items() if key.startswith("FILE_NAME_BAND")]

    
    # Attempt to stack the TOA reflectance rasters 
    # Note that we only process the first 7, since 8 onwards are specialty bands that we don't
    # really need

    # Get profile from initial band (doesn't matter which band, metadata for bands 1-7 are the same)
    init_src = rio.open(filenames[0])
    profile = init_src.profile

    # We are only stacking the first 7 bands
    profile.update(dtype = np.float32, nodata = np.nan, count = 7)

    # Test writing a masked array
    # Yep, writing out a masked array works perfectly fine, nodata are written as specified in the 
    # profile
    dst = rio.open(
        f"{metadata['PRODUCT_CONTENTS']['LANDSAT_PRODUCT_ID']}_TOA_STACKED.tif",
        "w",
        **profile
    )

    for band in tqdm(filenames):
        band_num = int(band.split(".")[0][-1])

        if band_num > 7:
            break

        src = rio.open(band)
        raster = src.read(1, masked = True) # Need to get the 2D array

        # filenames[0].split(".")[0][-1] gives band number from filename
        toa_ref = toa_reflectance(raster, band_num, metadata)

        # Remember to save as np.float32 for efficiency
        dst.write(toa_ref.astype(np.float32), band_num)

    dst.close()
    src.close()
    
    ### FOR THE SPECIALTY BANDS ###

    for band in tqdm(filenames[7:9]):
        band_num = int(band.split("B")[1].split(".")[0])

        src = rio.open(band)
        profile = src.profile
        profile.update(dtype = np.float32, nodata = np.nan)

        raster = src.read(1, masked = True) # Need to get the 2D array

        dst = rio.open(
            f"{metadata['PRODUCT_CONTENTS']['LANDSAT_PRODUCT_ID']}_BAND_{band_num}_TOA.tif",
            "w",
            **profile
        )

        # filenames[0].split(".")[0][-1] gives band number from filename
        toa_ref = toa_reflectance(raster, band_num, metadata)

        # Remember to save as np.float32 for efficiency
        dst.write(toa_ref.astype(np.float32), 1)
        src.close()
        dst.close()


# In[55]:





# In[56]:





# In[58]:


# src = rio.open("LC08_L1TP_113037_20230502_20230509_02_T1_BAND_8_TOA.tif")
# pan = src.read()

# pan_stretched = histogram_stretch(pan)

# show(pan_stretched)


# In[ ]:





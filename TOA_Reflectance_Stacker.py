#!/usr/bin/env python
# coding: utf-8

# In[4]:


import rasterio as rio
import geopandas as gpd
from rasterio.plot import show
import rasterio.mask
import os, fnmatch
import re
from tqdm import tqdm
import numpy as np
from rasterio.enums import Resampling
import sys
import shapely

### Command Line Issues ###


# In[11]:


### FUNCTIONS NEEDED ###

def parse_mtl(mtl_file):
    """
    Parses the landsat metadata file into a dictionary of dictionaries.
    
    Dictionary is split into several sub-dicts including PRODUCT_CONTENTS and IMAGE_ATTRIBUTES
    """
    
    with open(mtl_file) as f:
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

# in some cases we need to up or downsample
# https://rasterio.readthedocs.io/en/stable/topics/resampling.html

def resample_tif(raster_file, target_height, target_width):
    """
    given a raster file and a height/width with the same aspect ratio, 
    
    output a masked 2D array of resampled data
    """
    # we need to resample the land_use geotiff because it has a 10m scale
    with rio.open(raster_file) as dataset:

        # resample data to target shape
        data = dataset.read(
            out_shape=(
                dataset.count,
                target_height, # height and width of the up/downsampled tiff - this will force the 
                target_width  # opened landuse dataset into this shape
            ),
            resampling=Resampling.nearest, # nearest is good for land use, use cubicspline for DEM
            masked = True
        )

        # scale image transform object
        transform = dataset.transform * dataset.transform.scale(
            (dataset.width / data.shape[-1]),
            (dataset.height / data.shape[-2])
        )

        resampled_profile = dataset.profile
        resampled_profile.update(transform = transform, 
                       width = data.shape[-1], 
                       height = data.shape[-2])

    dataset.close()
    
    # data = masked numpy array, squeeze removes any dimensions of length 1 (i.e., 3D array with only
    # one stack will be converted into a 2D array)
    return data.squeeze(), resampled_profile


# In[16]:


if __name__ == "__main__":
    ### RETRIEVE ALL LANDSAT FILENAMES IN THIS METADATA ###
    # first argument should be the filename of the MTL metadata to be parsed
    try:
        metadata = parse_mtl(sys.argv[1])
    except:
        print("Metadata could not be read.")
#         metadata = parse_mtl("LC08_L1TP_113037_20230502_20230509_02_T1_MTL.txt")

    filenames = [value for key, value in metadata["PRODUCT_CONTENTS"].items() if key.startswith("FILE_NAME_BAND")]

    
    # Attempt to stack the TOA reflectance rasters 
    # Note that we only process the first 7, since 8 onwards are specialty bands that we don't
    # really need

    # Get profile from initial band (doesn't matter which band, metadata for bands 1-7 are the same)
    # Added os.path stuff to try and make it so that the script searches the directory in which
    # MTL file is located, i.e. - you don't need to put the .py script into the same directory 
    # as the MTL file
    init_src = rio.open(os.path.join(os.path.dirname(sys.argv[1]), filenames[0]))
    profile = init_src.profile

    # We are stacking the 10 bands (all bands except pan)
    profile.update(dtype = np.float32, nodata = np.nan, count = 10)

    # Test writing a masked array
    # Yep, writing out a masked array works perfectly fine, nodata are written as specified in the 
    # profile
    dst = rio.open(
        # the output stacked tifs SHOULD be output into the same directory as the .py script
        f"{metadata['PRODUCT_CONTENTS']['LANDSAT_PRODUCT_ID']}_TOA_STACKED.tif",
        "w",
        **profile
    )

    for band in tqdm(filenames):
        band_num = int(band.split("B")[1].split(".")[0])
        
        band = os.path.join(os.path.dirname(sys.argv[1]), band)

        # skip the panchromatic band since it doesn't add more information
        if band_num == 8:
            continue
        
        # need to catch thermal bands
        if band_num > 9:
            # thermal bands have lower resolution, must resample
            # NB strange, but my data appears to have higher reso thermal than normal
            resampled, resampled_transform = resample_tif(band, profile["height"], profile["width"])
            
            # no TOA reflectance for thermal bands, must normalize between 0-1 manually
            # possibly do this during pre-processing stage
            # resampled = resampled / 65535
            # -1 band num since band 8 is not inserted
            dst.write(resampled.astype(np.float32), band_num-1)
            
            # restart loop
            continue
        
        src = rio.open(band)
        raster = src.read(1, masked = True) # Need to get the 2D array
        src.close()

        # filenames[0].split(".")[0][-1] gives band number from filename
        toa_ref = toa_reflectance(raster, band_num, metadata)

        # Remember to save as np.float32 for efficiency
        if band_num < 8:
            dst.write(toa_ref.astype(np.float32), band_num)
        # need to -1 band_num because we drop band 8
        else:
            dst.write(toa_ref.astype(np.float32), band_num-1)
    
    print("saving...")
    dst.descriptions = tuple(['Coastal Aerosol', 
                              'Blue', 
                              'Green',
                              'Red',
                              'NIR',
                              'SWIR-1',
                              'SWIR-2',
                              'Cirrus',
                              'TIRS-1',
                              'TIRS-2'])
    dst.close()
    init_src.close()
    
    # If a studyarea shapefile has been provided
    if len(sys.argv) > 2:
        try:
            # shouldn't matter where the shapefile is located
            study_area = gpd.read_file(sys.argv[2])
        except:
            print("Shape file could not be read.")
#             study_area = gpd.read_file("studyarea_test.geojson")

        print("cropping...")
        
        xmin, ymin, xmax, ymax = study_area.total_bounds
        
        # Note that any existing stacked_cropped tif should be deleted before running
        command4 = 'gdalwarp -t_srs {crs} -te {x_min} {y_min} {x_max} {y_max} -r bilinear {src_file} {dst_file} -co COMPRESS=DEFLATE'
        os.system(command4.format(
            crs = f"EPSG:{study_area.crs.to_epsg()}", 
            x_min = np.floor(xmin),
            y_min = np.floor(ymin),
            x_max = np.ceil(xmax),
            y_max = np.ceil(ymax),
            src_file = f"{metadata['PRODUCT_CONTENTS']['LANDSAT_PRODUCT_ID']}_TOA_STACKED.tif",
            dst_file = f"{metadata['PRODUCT_CONTENTS']['LANDSAT_PRODUCT_ID']}_TOA_STACKED_CROPPED.tif"
        ))
    
        ### PREVIOUS ATTEMPT USING RASTERIO ###
#         print("cropping...")
#         src = rio.open(f"{metadata['PRODUCT_CONTENTS']['LANDSAT_PRODUCT_ID']}_TOA_STACKED.tif")
        
#         # Check if shapefile and raster crs are the same
#         if study_area.crs.to_epsg() == src.meta["crs"].to_epsg():
#             pass
#         else:
#             # to_crs does not work in place - following will update study_area to the correct crs
#             study_area = study_area.to_crs(f"EPSG:{src.meta['crs'].to_epsg()}")
            
#         # This study_area.total_bounds gives the xmin ymin xmax ymax of the geopandas df in an array
#         # Use it to create a box that we will pass the mask
#         polygon = shapely.geometry.box(*study_area.total_bounds)
        
#         # If you come across a weird error called rasterio has no attribute mask, import rasterio.mask
#         # Output is a tuple continue cropped raster + transform information
#         # Remeber that rio.mask.mask expects an iterable even if you're only passing in one polygon
#         cropped_raster, cropped_transform = rio.mask.mask(src, [polygon], crop = True)
        
#         # Update the profile used to save the new raster
#         profile = src.profile
#         profile.update(height = cropped_raster.shape[1], 
#                        width = cropped_raster.shape[2],
#                        transform = cropped_transform)

#         cropped_dataset = rio.open(f"{metadata['PRODUCT_CONTENTS']['LANDSAT_PRODUCT_ID']}_TOA_STACKED_CROPPED.tif", "w", **profile)
#         # Also remember to convert the image into int16, as specified in the profile
#         cropped_dataset.write(cropped_raster)
#         cropped_dataset.descriptions = tuple(['Coastal Aerosol', 
#                           'Blue', 
#                           'Green',
#                           'Red',
#                           'NIR',
#                           'SWIR-1',
#                           'SWIR-2',
#                           'Cirrus',
#                           'TIRS-1',
#                           'TIRS-2'])
#         cropped_dataset.close()
        

    
#     ### FOR THE SPECIALTY BANDS ###

#     for band in tqdm(filenames[7:9]):
#         band_num = int(band.split("B")[1].split(".")[0])

#         src = rio.open(band)
#         profile = src.profile
#         profile.update(dtype = np.float32, nodata = np.nan)

#         raster = src.read(1, masked = True) # Need to get the 2D array

#         dst = rio.open(
#             f"{metadata['PRODUCT_CONTENTS']['LANDSAT_PRODUCT_ID']}_BAND_{band_num}_TOA.tif",
#             "w",
#             **profile
#         )

#         # filenames[0].split(".")[0][-1] gives band number from filename
#         toa_ref = toa_reflectance(raster, band_num, metadata)

#         # Remember to save as np.float32 for efficiency
#         dst.write(toa_ref.astype(np.float32), 1)
#         src.close()
#         dst.close()


# In[2]:


# import rasterio as rio
# from rasterio.plot import show

# src = rio.open("LC08_L1TP_113037_20230502_20230509_02_T1_TOA_STACKED_CROPPED.tif")

# # test = src.read()

# # test.shape

# show(src.read([4,3,2]))


# In[3]:


# study_area = gpd.read_file("studyarea_test.geojson")
# src = rio.open(f"LC08_L1TP_113037_20230502_20230509_02_T1_TOA_STACKED.tif")

# # Check if shapefile and raster crs are the same
# if study_area.crs.to_epsg() == src.meta["crs"].to_epsg():
#     pass
# else:
#     # to_crs does not work in place - following will update study_area to the correct crs
#     study_area = study_area.to_crs(f"EPSG:{src.meta['crs'].to_epsg()}")

# # This study_area.total_bounds gives the xmin ymin xmax ymax of the geopandas df in an array
# # Use it to create a box that we will pass the mask
# polygon = shapely.geometry.box(*study_area.total_bounds)

# # If you come across a weird error called rasterio has no attribute mask, import rasterio.mask
# # Output is a tuple continue cropped raster + transform information
# # Remeber that rio.mask.mask expects an iterable even if you're only passing in one polygon
# cropped_raster, cropped_transform = rio.mask.mask(src, [polygon], crop = True)

# # Update the profile used to save the new raster
# profile = src.profile
# # Note that height/width are elements 1 and 2 for a 3D raster
# profile.update(height = cropped_raster.shape[1], 
#                width = cropped_raster.shape[2],
#                transform = cropped_transform)

# cropped_dataset = rio.open(f"cropped_test.tif", "w", **profile)
# # No need to specify a band number if you are writing a multi band raster
# cropped_dataset.write(cropped_raster)
# cropped_dataset.descriptions = tuple(['Coastal Aerosol', 
#                               'Blue', 
#                               'Green',
#                               'Red',
#                               'NIR',
#                               'SWIR-1',
#                               'SWIR-2',
#                               'Cirrus',
#                               'TIRS-1',
#                               'TIRS-2'])
# cropped_dataset.close()


# In[7]:


# src = rio.open("cropped_test.tif")

# show(src.read([4,3,2]))

# # show(cropped_raster[[3,2,1],:,:])


# In[42]:


# show(cropped[0][[3,2,1],:,:], transform = cropped[1])


# In[30]:


# src.bounds


# In[ ]:





# In[ ]:


### consider adding some code to crop the multiband raster into study area
# based on this gdal call:
# gdalwarp -t_srs EPSG:3098 -te 547579 3600677 631939 3667730 
# "C:\..\LC08_L1TP_113037_20230502_20230509_02_T1_TOA_STACKED.tif" 
# nagasaki_toa.tif


# In[5]:


# metadata = parse_mtl("LC08_L1TP_113037_20230502_20230509_02_T1_MTL.txt")


# In[20]:


# src = rio.open("LC08_L1TP_113037_20230502_20230509_02_T1_B10.TIF")
# test = src.read()

# # print(np.max(test, axis = (1,2)))
# # print(np.min(test, axis = (1,2)))
# # src.profile

# show(test[:, 5000:6000, 3000:5000])


# In[58]:


# src = rio.open("LC08_L1TP_113037_20230502_20230509_02_T1_BAND_8_TOA.tif")
# pan = src.read()

# pan_stretched = histogram_stretch(pan)

# show(pan_stretched)


# In[ ]:





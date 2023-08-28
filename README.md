# Basic Geospatial Workflow

This repository contains the code that I have used to determine the land use distribution of Nagasaki, Japan using LANDSAT data.

This workflow includes generating top-of-atmosphere (TOA) reflectance data from LANDSAT images, cropping raster images to a specified study area, creating a shape file consisting of polygons used to train the supervised machine learning model, training both supervised and unsupervised machine learning models, as well as visualizing the resultant data.

The workflow used here can be applied to any collection of data downloaded from the USGS Earth Explorer website. 

To do:
 - Add code to TOA_reflectance_stacker.py to crop the output raster into the study area based on a shapefile.
 - Add animations and summary statistics to show land use change over time.

## Data Collection

LANDSAT data was downloaded from the [USGS earth explorer website](https://earthexplorer.usgs.gov/). In addition to LANDSAT data, this database contains a variety of satellite information such as MODIS land use data. Note that LANDSAT files are quite large. The LANDSAT products provided include 11 bands across the electromagnetic spectrum. Bands 1–7 contain reflectance information, while band 8 is a panchromatic image that takes information from bands 2 to 5. Band 9 contains cirrus data usually used for quality control, while bands 10 and 11 contain thermal emissivity data. 

| Bands                                | Wavelength<br>(micrometers) | Resolution<br>(meters) |
| ------------------------------------ | --------------------------- | ---------------------- |
| Band 1 - Coastal aerosol             | 0.43-0.45                   | 30                     |
| Band 2 - Blue                        | 0.45-0.51                   | 30                     |
| Band 3 - Green                       | 0.53-0.59                   | 30                     |
| Band 4 - Red                         | 0.64-0.67                   | 30                     |
| Band 5 - Near Infrared (NIR)         | 0.85-0.88                   | 30                     |
| Band 6 - Shortwave Infrared (SWIR) 1 | 1.57-1.65                   | 30                     |
| Band 7 - Shortwave Infrared (SWIR) 2 | 2.11-2.29                   | 30                     |
| Band 8 - Panchromatic                | 0.50-0.68                   | 15                     |
| Band 9 - Cirrus                      | 1.36-1.38                   | 30                     |
| Band 10 - Thermal Infrared (TIRS) 1  | 10.6-11.19                  | 100                    |
| Band 11 - Thermal Infrared (TIRS) 2  | 11.50-12.51                 | 100                    |

Different combinations of LANDSAT bands can be used to highlight different spectral signatures, which can, in turn, be used to identify specific properties or characteristics about a field of view. A discussion of the possible combinations of LANDSAT bands is out of the scope of this Readme.

## Calculating TOA Reflectance

It should be noted that, in LANDSAT Level 1 products, Bands 1 through 9 are represented in the form of quantized and calibrated Digital Numbers (DN) representing the multispectral image data ranging from 1–65536. These DN numbers should be converted to TOA reflectance, which is a physical quantity representing the ratio of incident to reflected radiation. The specific formulas used to calculate this TOA reflectance can be found [here]( https://www.usgs.gov/landsat-missions/using-usgs-landsat-level-1-data-product). Most importantly this conversion requires information from the MTL metadata text file packaged with each LANDSAT product.

TOA_Reflectance_Stacker.py can be run from the command line using the following syntax: `TOA_Reflectance_Stacker.py mtl_file`. I strongly recommend running this file using the OSGeo4W PowerShell as future releases may use GDAL CLI commands.

Running this command creates a stack of raster data from LANDSAT bands 1 to 11. A notable exception is that band 8 is not included in the stack, since the panchromatic band does not contain any additional information that is not already present in bands 2 through 5. The primary advantage of the panchromatic band is that it has a much higher resolution than the rest of the bands. Since it does not provide additional information for land use classification and because the higher resolution means that its size is much larger than the other bands it was excluded from the band stack. 

In addition, please note that while TOA reflectance is calculated for bands 1 through 9, and consequently has a value between 0 and 1, bands 11 and 12 contain emissivity data which does not have TOA reflectance equivalent. 



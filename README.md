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

TOA_Reflectance_Stacker.py is a script that uses metadata from the LANDSAT MTL file to calculate the TOA of bands 1 through 9 and stacks them together with bands 11 and 12 into a 10-band multiband raster.A notable exception is that band 8 is not included in the stack, since the panchromatic band does not contain any additional information that is not already present in bands 2 through 5. The primary advantage of the panchromatic band is that it has a much higher resolution than the rest of the bands. Since it does not provide additional information for land use classification and because the higher resolution means that its size is much larger than the other bands it was excluded from the band stack. 

In addition, please note that while TOA reflectance is calculated for bands 1 through 9, and consequently has a value between 0 and 1, bands 11 and 12 contain emissivity data which does not have TOA reflectance equivalent - these values thus still range between 1-65535. 

The script has been set up such that you only need to provide the path to the MTL file and ensure that it is in the same folder as the rest of the LANDSAT bands. The output TOA multiband raster will be generated into the folder that the script is located in.

In addition, the script has been updated to allow you to crop the multiband raster to a smaller study area using a shapefile. This shapefile can be in any file format that can be read by Geopandas, such as ESRI shapefiles and GeoJSONs. Note that the raster will be cropped to the maximum rectangular extent of the shapefile. Similarly the cropped raster will be output into the folder where the script is located. 

One important thing to note is that this script makes use of the gdal command line; Another words your environment must have gdal installed. If you have an Anaconda distribution this is relatively easy by creating a new environment (`conda create --name env_name`) and running `conda install -c conda-forge gdal`.

The script can be run from the command line using the following syntax: `TOA_Reflectance_Stacker.py mtl_file study_area_shapefile`. Note that adding the study area shape file is optional but highly recommended as the full multiband raster can be relatively large (~1 GB).

## Data Preprocessing

The first thing that should be done is to crop the stacked raster image to your study area. I accomplished this using the following GDAL CLI command: `gdalwarp -t_srs target_crs -te xmin ymin xmax ymax source_filepath destination_filepath`. This command crops the raster image into the stated extent as well as converts the CRS of the image to the target CRS. Remember that all files used in the project should use the same CRS. Future releases may consider integrating this step into TOA_reflectance_stacker.py. 

The stacked image can now be used for a variety of visualizations, as well as for the calculation of several indices, such as NDVI. It should be noted however, that without additional processing, the resultant images are of extremely low contrast. TOA_reflectance_stacker.py contains a function called `histogram_stretch` That does a simple linear histogram stretch that allows you to plot the raster and generate much more visually informative images.

![alt text](https://github.com/Pinnacle55/nagasaki-ml/blob/a77e70fd0860aabdc0d1b1427b4bfe8304e24d83/without_histogram_stretch.jpg?raw=true "Without stretching")

![alt text](https://github.com/Pinnacle55/nagasaki-ml/blob/a77e70fd0860aabdc0d1b1427b4bfe8304e24d83/with_histogram_stretch.jpg?raw=true "With stretching")

## Machine Learning: Unsupervised

We can now start with some basic machine learning. Unsupervised machine learning can be conducted without user-labeled data. The user needs only to select a small sub area of the study in which the model should be trained. This sub area should have clear examples of all of the different classes that you would like the model to identify. 

I selected the following sub area because it exhibits several prominent classes across the data set including forested, marine, and urban areas.

![alt text](https://github.com/Pinnacle55/nagasaki-ml/blob/82923b7ae834f8d45cba5a90199ad19692de575e/Images/Study%20Area.png?raw=true "Unsupervised Study Area")

## Training Data

At this point, we need to generate training data for our land use classifier. Ideally you should find land use data from local governments or elsewhere online. In cases where this data is absent, you can generate your own land use data. For example, I generated some training data in the form of a shapefile drawn in a GIS software (QGIS). I drew polygons around areas of “known” land use by reference to the LANDSAT images as well as Google Earth. In my case, I used a very basic classification scheme consisting of only four different types of land use: water, urban, forest, and cropland. 

It is important to ensure that you identify areas of the same class but with relatively different spectral signatures. For example, deep water in the ocean and shallow water filled with sediment are both in the “water” class but have very different spectral signatures because of the way sediment interacts with the different LANDSAT bands. It is important to ensure that you create training data polygons that account for both of these possibilities to prevent misclassifications.

Once you have finished with your classifications, save the shapefile as a GeoJSON. Remember to set the correct CRS when saving the shapefile.

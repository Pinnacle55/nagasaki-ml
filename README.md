# Basic Geospatial Workflow

This repository contains the code that I have used to determine the land use distribution of Nagasaki, Japan using LANDSAT data.

This workflow includes generating top-of-atmosphere (TOA) reflectance data from LANDSAT images, cropping raster images to a specified study area, creating a shape file consisting of polygons used to train the supervised machine learning model, training both supervised and unsupervised machine learning models, as well as visualizing the resultant data.

The workflow used here can be applied to any collection of data downloaded from the USGS Earth Explorer website. 

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

Different combinations of LANDSAT bands can be used to highlight different spectral signatures, which can, in turn, be used to identify specific properties or characteristics about a field of view. A discussion of the possible combinations of LANDSAT bands is out of the scope of this Readme, but can be easily found online.

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

NEW: I have recently developed a piece of code named `landsat_processing_utils.py` which contains a set of functions that are commonly used in the preprocessing of landsat data. It utilises a variety of geospatial Python libraries including rasterio, geopandas, earthpy, and shapely. In particular, it is capable of combining the TOA reflectance calculation, cropping, and stacking into a single command.

The functions available on the command line are shown here:
```
On Command Line
>>> python landsat_processing_utils.py --help
usage: landsat_processing_utils.py [-h] [-m MASK] [-s] [-c] [-o OUTDIR] [-d] filepath

Calculates TOA reflectance for Level 1 Landsat products. Additional flags can be set to crop, stack, and ignore sun
elevation correction.

positional arguments:
  filepath              Path to folder containing all Landsat bands and MTL file.

options:
  -h, --help            show this help message and exit
  -m MASK, --mask MASK  Provide a shapefile that the images will be cropped to
  -s, --stack           Create a stacked raster of the images - default False
  -c, --sun_corr        add this flag if you DON'T want to do a sun elevation correction - default True
  -o OUTDIR, --outdir OUTDIR
                        Specify an output folder to save TOA corrected images
  -d, --cleanup         If cropping, choose whether to delete the uncropped TOA images - default False
```

Additional functionality included within the module itself includes the parsing of level 2 product QA bands for cloud detection and the manipulation of raster data in preparation for machine learning.

## Machine Learning: Unsupervised

We can now start with some basic machine learning. Unsupervised machine learning can be conducted without user-labeled data. The user needs only to select a small sub area of the study in which the model should be trained. This sub area should have clear examples of all of the different classes that you would like the model to identify. 

I selected the following sub area because it exhibits several prominent classes across the data set including forested, marine, and urban areas.

![alt text](https://github.com/Pinnacle55/nagasaki-ml/blob/82923b7ae834f8d45cba5a90199ad19692de575e/Images/Study%20Area.png?raw=true "Unsupervised Study Area")

The first thing we need to do when conducting unsupervised learning is to decide how many classes there are in the dataset. This is a classic problem of unsupervised machine learning: additional classes will reduce the number of misclassifications, but we'll take additional resources and will have a higher chance of overfitting.

We use a simple elbow method to determine the number of classes in the dataset. The elbow method plots the error or distortion against the number of clusters that are predicted. A kink in the graph indicates a break point at which additional classes have a diminishing effect on the accuracy of the model.

![alt text](https://github.com/Pinnacle55/nagasaki-ml/blob/32e362ca2b1c60a27f86cc135f36a859dff8e749/Images/Elbow%20Method.png?raw=true "Elbow Method")

Based on the elbow method, we see that instantiating a clustering algorithm with three or five classes would be ideal for this dataset. I instantiated a kmeans model with five clusters and trained it on the study area, leading to the following:

![alt text](https://github.com/Pinnacle55/nagasaki-ml/blob/528f4246749a4c0dfe30471e6ce9563c3a7d30b5/Images/Unsupervised%20KMeans%20Model%20-%2020230903.png?raw=True "Unsupervised Learning")

The model has actually done quite well for itself, being able to clearly identify water regardless of sediment content. It has also been able to identify urban areas as well as cropland areas. It is also seemingly been able to do to identify two different types of forested areas - however, these differences are likely just to do with differences in terrain relief and aspect rather than any form of land use.

However, there are some clear issues when attempting to extend our model to the rest of the scene. 

![alt_text](https://github.com/Pinnacle55/nagasaki-ml/blob/f1e7f71844ae8fd74703aa3701c8e85cdfcc7db7/Images/unsupervised_5cluster.png)

Specifically, there are some clear issues with the identification of sediment-laden water - this can be mostly seen in the northeast section of the scene, where water that has a high sediment content has been misclassified as forest. In addition, there is a cove near the middle of the scene that has been misclassified as an urban area.

One way of attempting to ameliorate this issue is to select a study area that contains all of the possible different land use types that can be found in the scene. Although we selected a very good training area, it did not contain any water bodies with a significant amount of sediment; we'll need to pick an area that contains this information. We can then train our unsupervised model on this new study area - in addition, we will increase the number of clusters to six in order to attempt to account for the sediment-laden water.

![alt_text](https://github.com/Pinnacle55/nagasaki-ml/blob/3e83002d746c689f8f1f5bec8e4d8e8b826f9d6b/Images/unsupervised_6cluster_training.png)

When we extend this model to the rest of the scene, we find that the model now correctly predicts the presence of sediment laden water in the appropriate areas.

![alt_text](https://github.com/Pinnacle55/nagasaki-ml/blob/3e83002d746c689f8f1f5bec8e4d8e8b826f9d6b/Images/unsupervised_beststudyarea_6cluster.png)

### Additional Tweaks

Another way of addressing the issue is to crop areas of the image that are not useful for the analysis that we are trying to conduct. For example, if we are only interested in land use, we may wish to crop the image such that only the land areas are being analysed (i.e., we can remove all water bodies from the training set). We can do this by cropping the image to our administrative boundary dataset. The image below shows an unsupervised learning model with three clusters trained on only the areas within the municipal boundaries of Nagasaki.

![alt_text](https://github.com/Pinnacle55/nagasaki-ml/blob/3e83002d746c689f8f1f5bec8e4d8e8b826f9d6b/Images/cropped_unsupervised_3cluster.png)

Of course, we can extend this workflow to a four-cluster unsupervised learning model and attempt to extend the model to other scenes in the data set in order to get a time series visualisation of the study area during the year 2021.

![alt_text](https://github.com/Pinnacle55/nagasaki-ml/blob/3e83002d746c689f8f1f5bec8e4d8e8b826f9d6b/Images/unsupervised_landuse_seasonal.gif)

We can see that the unsupervised learning model has worked quite well. In particular, this visualisation has very clearly shown how the cropland changes with the seasons: specifically, you can see how the cropland changes from being classified as urban/cropland during winter to being classified as forest_1 in the summer. Machine learning models tend to struggle with the classification of cropland, especially in time series data, because the spectral signature of cropland changes dramatically over the course of a year.

## Training Data for Supervised Learning

At this point, we need to generate training data for our land use classifier. Ideally you should find land use data from local governments or elsewhere online. In cases where this data is absent, you can generate your own land use data. For example, I generated some training data in the form of a shapefile drawn in a GIS software (QGIS). I drew polygons around areas of “known” land use by reference to the LANDSAT images as well as Google Earth. In my case, I used a very basic classification scheme consisting of only four different types of land use: water, urban, forest, cropland, and bare land. Naturally, you can create as many classifications as you would like.

It is important to ensure that you identify areas of the same class but with relatively different spectral signatures. For example, deep water in the ocean and shallow water filled with sediment are both in the “water” class but have very different spectral signatures because of the way sediment interacts with the different LANDSAT bands. It is important to ensure that you create training data polygons that account for both of these possibilities to prevent misclassifications.

Once you have finished with your classifications, save the shapefile as a GeoJSON. Remember to set the correct CRS when saving the shapefile. The image below shows the location of my training polygons in comparison to the rest of the image.

![alt_text](https://github.com/Pinnacle55/nagasaki-ml/blob/2eaa2f83463becdd7389bf3da544efed96777b05/Images/training_polygons.png)

The next step is to prepare the training data in an appropriate format. This is actually more tricky than it looks because you need to use rasterio’s mask function in order to select only the rest of data that is covered by the training polygons. Notably, the `rio.mask.mask()` function only takes in a raster file in ‘read’ mode – it cannot be used for loaded rasters. 

### Cross-Validation and Pipelines

At this point, we can start using some more advanced machine learning techniques to improve the results of our algorithms. In particular, because we have ground truth data that we can compare our predictions to, we can start using cross-validation and accuracy metrics in order to assess the effectiveness of our machine learning models.

Cross validation refers to the act of splitting the training data into parts, training the model on one of those parts and validating the model against the other parts. This allows us to get a good idea of whether the model works well across the training data. It is also important to select an appropriate accuracy metric by which to assess our model: in most cases, the accuracy metric is not particularly useful - we may want to use more specific metrics for our use cases such as F1 scores or log loss. Your choice of metrics will depend on the kind of error that you are trying to minimise.

In addition, we can use some of the workflow aids that are available in scikit-learn, such as pipelines. Pipelines allow you to run a set of data through standardised workflow so that you do not need to call multiple types of functions each time you want to train and test a set of data. Although each pipeline can only use one estimator, you can call as many preprocessing functions as you wish, such as principal component analysis (PCA) and/or scaling.

An example of how you can set up an Naive-Bayes estimator that scales the data and cross validates the model is shown below.

```
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

gnb_clf = make_pipeline(StandardScaler(), GaussianNB())

# 5 fold cross-validation
cross_val_score(gnb_clf, X_train, y_train, cv = 5, scoring='f1_macro')

>>> array([0.72360339, 0.89682849, 0.80111729, 0.53799032, 0.75659121])
```

![alt_text](https://github.com/Pinnacle55/nagasaki-ml/blob/2eaa2f83463becdd7389bf3da544efed96777b05/Images/supervised_GNB.png)

![alt_text](https://github.com/Pinnacle55/nagasaki-ml/blob/2eaa2f83463becdd7389bf3da544efed96777b05/Images/supervised_GNB_studyarea.png)

The Naïve-Bayes model has done a relatively good job with the classification of land use across the study area. There were no issues with sediment-laden water like we saw in the unsupervised learning model, and even in the case of deltas they were correctly identified as bare land. However, there are still some clear issues that can be observed: for example, the model clearly overestimates the amount of bare land in the scene - this is particularly clear in the smaller study area, where a significant amount of the urban area has been misclassified as bare land.

### Hyperparameter Tuning

You can, of course, experiment with different types of models. In addition, you can attempt to use hyperparameter tuning to improve the predictive power of the model. Naïve-Bayes model don’t have any meaningful parameters can be tuned, so let’s use a LinearSVC model for our next attempt (note: we use a LinearSVC model because `SVC()` models cannot be effectively extended to extremely large datasets. Since we are dealing with image data with millions of pixels, we use a `LinearSVC()` model for greater efficiency.)

Hyper parameter tuning refers to tuning several parameters of the machine learning model in order to identify the model that has the best performance. This can be done using a variety of techniques: in `sklearn`, this can be done with GridSearchCV, which allows you to specify a dictionary of hyperparameters which the function will then test to identify the best performing model. The code snippet below shows the set-up of a hyperparameter tuning workflow.

```
from sklearn.model_selection import RandomizedSearchCV

# dictionary or list of dictionaries showing the hyperparameters to be tuned
# need to use the clf__ prefix to distinguish between multiple parts of the pipeline
# for example, if you are using PCA parameters or TDIFD vectorizers
parameter_grid = [
    {
        'linearsvc__penalty': ['l1'], 
        'linearsvc__C': [1, 10, 100, 1000]
    },
    {
        'linearsvc__penalty': ['l2'],
        'linearsvc__C': [1, 10, 100, 1000], 
        'linearsvc__loss': ['hinge', 'squared_hinge']
    }
]

# This does a random cross-validated search of the parameter grid
# if you don't specify cv, automatically uses cv = 5
# should actually use GridSearchCV for an exhaustive search of the grid
random_svc_search = RandomizedSearchCV(
    estimator=svc_pipeline,
    param_distributions=parameter_grid,
    n_iter=40,
    random_state=0,
    n_jobs=2,
    verbose=1,
    scoring = ('accuracy', 'f1_macro'), # list of scorers that the cv search should be scored by
    refit = 'f1_macro'                  # if using multiple scorers, refit sets which scorer should be used for best_params
)
```

Once the search has been completed, you can read in the results of the search into a pandas DataFrame in order to inspect the results of the search (`pd.DataFrame.from_dict(random_svc_search.cv_results_, orient = "columns")`). If you just want to identify the best model then `random_cv_search.best_estimator_` will give you the details of the best performing model as identified in the search. You can run the predict function directly on the RandomSearchCV object as shown below:

```
y_pred = random_svc_search.predict(X_test) 
```

![alt_text](https://github.com/Pinnacle55/nagasaki-ml/blob/2eaa2f83463becdd7389bf3da544efed96777b05/Images/supervised_linearsvc.png)

![alt_text](https://github.com/Pinnacle55/nagasaki-ml/blob/2eaa2f83463becdd7389bf3da544efed96777b05/Images/supervised_linearsvc_studyarea.png)

A cursory analysis shows that this is by far the best performing model that we have. There are no issues with sediment-laden water or with the over-prediction of bare land.

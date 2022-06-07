# Final Project
### By: Carly Schippits, Shai Slotky, Merritt Smith, and Sophia Mlawer


## Background
Deforestation affects us all, whether we realize it or not. According to the WWF, around 15 billion trees are now being cut down every year across the world. That rate isn't sustainable for people, wildlife, and the climate.
 
Forests are very important for the health of the planet. They provide food and shelter for so many varieties of species. Additionally, forests have a big influence on rainfall patterns, water and soil quality, and flood prevention. With the growing threat of global warming, deforestation is even more important to predict. Forest loss and damage is the cause of around 10% of global warming. Trees absorb and store carbon dioxide and if forests are cleared, or even distributed, they release carbon dioxide and other greenhouse gases [1]. Therefore, stopping deforestation and fighting against the climate crisis go hand in hand.
 
Specifically, we focused our efforts on predicting deforestation in the Amazon rainforest in Brazil. The Amazon rainforest contains half of the planet's remaining tropical forests and holds 10% of the global carbon reserves [2]. This makes it very important to climate efforts both locally and around the world. Recently, there has been increased economic pressure and weakening of environmental agencies and legislation by the Brazilian government that makes tracking and predicting deforestation even more important. Traditional stakeholders, like non-profits and environmental groups, have limited resources and if we are able to help them know where deforestation is likely to happen next, they can focus their resources and efforts on those areas. 

## Overview of the Pipeline
### Creating the Data
We decided to use the Terra-i deforestation data as our dataset [3]. Terra-i uses satellite imagery from NASA’s MODIS (Moderate Resolution Imaging-Spectrometer) and TRMM (Tropical Rainfall Measuring Mission) sensors to detect tree cover loss through a computational algorithm designed to detect changes of greenness over time and to relate them to rainfall.

Terra-I offers .tif files that represent large sections of the globe. We chose to build out our prototype model with only one of these sections/tiles, but with 18 years of data for that tile (the data is observed on an annual basis). We chose to use the tile that represents the State of Amazonas in Brazil due to the identifiable trends we noticed when initially visualizing the data and due to the prevalence of the area in narratives around deforestation globally. Our data processing implementation is scalable beyond a single tile, but the current model may not be externally valid beyond this region since the deforestation largely takes place immediately next to the Amazon.  

Each of the raw data rasters represents a 10km$^2$ swath of land. Each raster contains 40x40 pixels. Each pixel contains an integer value: -9999 representing missing data (for instance if the pixel is entirely water or mountains), 0 if no deforestation happened in that pixel in that year, or a number between 1 and 365 representing the day on which the deforestation was detected. We are not interested in predicting the exact day on which a deforestation event occurs, so we masked any positive values to 1. We also masked all missing values to have a more efficient data representation.

To create our training data, we used Ray. For each raster, we read all 18 large tif files into memory as NumPy arrays, then exploited the fact that each one has the same dimensions to iterate over all of them simultaneously. We chose to make our dataset be all tif files of size 40x40, because that represents a 10km$^2$ area. Ray allows us to parallelize this so that we don't have to copy each NumPy array for every separate process. We can instead put the NumPy arrays into a standard Python list, and call ray.put() on it to get a reference to the list of arrays---much like you would be able to in a lower level language like C++ or Go. This set-up allowed us to write tens of thousands of tif files in minutes while parallelizing across multiple cores locally. At the end of this process, we have 18 data directories, each representing a year ranging from 2004 to 2021. Inside each of these directories are 2,297 tif files, each directory having files with the same names that represent the same observed area over 18 years. These tif files that we chose represent areas in which a significant amount of deforestation happened. We could easily increase the size of our dataset if we so chose with many more instances in which deforestation did not happen, but we were concerned about an unbalanced dataset because, for most pixels, deforestation does not happen.  

Once we had all the rasters, they were saved in an Amazon S3 bucket for easy access in our analysis. This allowed us to save them in the original format and have different objects for the multiple years of data. Additionally, this made it so that none of us had to store huge amounts of data on our personal computers and allowed us all to have easy access to the data.

The code for this step of the pipeline is located at [working_code/gen_tiles.py](working_code/gen_tiles.py). 

### Reading the Data
Now we are ready to read the data into Python for our analysis. This process is a prime candidate for parallelization because we want to read in tif files from different folders in the same way and concatenate the files together. To speed up this process, we used Dask and ImageIO to lazily load and concatenate the data into a single large Dask array with four dimensions (Number of samples x 18 years x 40 pixels x 40 pixel).

The code for this step of the pipeline is located at [working_code/dask_keras_gpu.ipynb](working_code/dask_keras_gpu.ipynb).

### Pre-Processing the Data
Our next step before estimating the model was to one-hot encode the dataset. This entailed adding a fifth dimension to our dataset containing three channels (one for each possible category: missing, not deforested, deforested). Because one-hot encoding is a simple task that needs to be performed on every pixel in our large dataset, it is a perfect operation to parallelize. We again used Dask to perform this pre-processing step.

The code for this step of the pipeline is located at [working_code/dask_keras_gpu.ipynb](working_code/dask_keras_gpu.ipynb).

### Building the Model
For the model, we chose a CNN-LSTM [4]. The CNN-LSTM incorporates both a spatial aspect (how deforestation in certain rasters affects predicted deforestation in nearby rasters) and a time aspect (how deforestation in past years affects deforestation in future years). For each raster, the model takes in a sequence of frames (each observed in a different year) and predicts the next raster in the sequence. This predicted raster has three channels, each containing a float between 0 and 1 which indicates the probability that a particular pixel will be missing data, will not be deforested, or will be deforested in the next frame. Using the model output, we can then take the maximum of the three probabilities for each pixel to make a prediction (missing, not deforested, or deforested) for that pixel.   

Because our deep learning model takes images as input, we decided that it would be a prime candidate for scaling up with GPUs. GPUs support better processing for high-resolution images by breaking down complex modeling tasks into smaller subtasks that can be concurrently performed. As such, we enabled our custom Keras model to run on GPUs instead of CPUs.

The code for this step of the pipeline is located at [working_code/dask_keras_gpu.ipynb](working_code/dask_keras_gpu.ipynb).

## Other Attempted Approaches
To load the data, we had initially tried using Rasterio in a Numba-enabled for loop. We ran into issues with this approach, as Numba is not compatible with Rasterio, OS (the package we used to grab the folder and tif names), and several of the NumPy functions that we were using to concatenate and reshape our arrays. As a result, we explored alternative parallel methods for loading the data, including RioXarray, before moving to Dask.

To parallelize our Keras model, we initially tried using Dask-ML's SciKeras wrapper. While the wrapper accepts custom Keras models, it only works with models that take in 1D or 2D inputs (not the 5D inputs that we are working with). We considered reshaping our input into fewer dimensions but decided that we could not do so without losing valuable information (i.e., spatial or time dimensions). Next, we investigated the possibility of using a custom optimizer but we discovered that this did not circumvent the dimensionality issue, which is an inherent limitation of the SciKeras wrapper. As a result, we decided to proceed with our original Keras model and run it on GPUs.

Code for other attempted approaches can be found in [other_attempted_solutions/](other_attempted_solutions/).

## Further Parallelization
If a researcher wanted to add a cross validation step to the model, they could do so and enable the cross validation to run on GPUs.

## Limitations
One limitation that we've briefly discussed above is the problem of external validity. Does our model predict well in contexts outside of the deep Amazon rainforest? There are good reasons to think not. The Amazon is a distinct ecosystem in which it is difficult to access anywhere that is not near water, so we would expect the vast majority of deforestation to happen near water. This is likely not true in other contexts. We also think that our model is limited by not taking into account the larger context beyond the 40x40 raster it sees. If a raster is just beyond the edge of the Amazon such that none of its pixels are missing data, but is very close by, this area is much more likely to be deforested than is an identical raster that is farther away from the water. A more thorough approach might take into account not only the time dimension as we do but also incorporate the larger context across rasters.

We might also be concerned about internal validity. There were massive wildfires in the Amazon in the past few years. Should these fires and the deforestation they caused be considered as part of the prediction, or a one-time anomalous effect that should be discarded? Population growth and global warming would suggest that they should be incorporated as part of the prediction model, but this model is insufficiently robust to capture all of these processes. 

## Responsibilities of Each Group Member
### Creating the data
Merritt found the data, downloaded it, and created all of the training data using Ray. This involved masking the data, chunking it out, creating the thousands of files per year, and uploading those files to S3. He looked into using standard Pandas multiprocessing and MPI, but found Ray most useful.

### Reading the Data
Carly wrote the unparallelized data loading code.

Shai and Sophia parallelized the data loading code using ImageIO and Dask. They also tried a number of other approaches to data loading such as (1) a Numba-enabled for loop using Rasterio and NumPy and (2) RioXarray.

### Preprocessing the Data
Carly wrote the unparallelized code to pre-process (one-hot encode) the dataset. 

Carly parallelized the data pre-processing using Dask. She also wrote a Numba-enabled pre-processing function, which we did not end up using because we switched from NumPy arrays to Dask arrays.

### Building the Model
Carly wrote the code for the Keras model.

Sophia and Shai parallelized the Keras model using Distributed training in Tensorflow via GPU. Shai and Sophia also researched methods about how to parallelize the Keras model including exploring how SkiKeras works on a typical Keras model. Shai and Carly attempted (unsuccessfully) to parallelize the Keras model using Dask-ML's SciKeras wrapper.

## References
[1] https://wwf.panda.org/discover/our_focus/forests_practice/deforestation_fronts_/

[2] https://iopscience.iop.org/article/10.1088/1748-9326/ac146a

[3] [Terra-i Data Portal](https://www.terra-i.org/terra-i/data.html).

[4] Our code for the CNN-LSTM was inspired by [this Keras example](https://keras.io/examples/vision/conv_lstm/).

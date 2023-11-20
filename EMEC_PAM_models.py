

### This script explores models to classify underwater acoustic data to 
### identify marine life



## ==================================================================
### a lot of information is gathered through research from the web

# one very useful website is from Mike Polinowski on Deep Audio analyses
# https://mpolinowski.github.io/docs/IoT-and-Machine-Learning/ML/2022-04-01-tensorflow-audio-classifier/2022-04-01/


### For this work we need tensorflow and GPU power

# if not already installed, create a new environment for tensorflow and
# install it
# conda create -name envname python=3.9 tensorflow
# pip install tensorflow-macos=2.13
# pip install tensorflow-metal=1.0.0

# also install all necessary modules for analyses, such as numpy, pandas etc.




######################################################################
# now lets start coding :)
# we already have a depository with tensorflow called ketos, following this 
# webpage on classifying whale calls in North America
# https://docs.meridian.cs.dal.ca/ketos/tutorials/create_database_simpler/index.html
# it is a very good tutorial on how to create training data and using a 
# classfifier model to identify whale calls



# import modules to jupyter notebook
import os
from matplotlib import pyplot as plt
import tensorflow as tf 





# Data Loading
# ========================


# To work with the recorded audio files we can select the corresponding data paths - in the example below 
# we pick both an example file that contains the signal we are looking for and one that only comes with background noise:

CAPUCHIN_FILE = os.path.join('data', 'positives', 'XC3776-3.wav')
NOT_CAPUCHIN_FILE = os.path.join('data', 'negatives', 'afternoon-birds-song-in-forest-0.wav')
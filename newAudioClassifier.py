
# New audio classifier

## code to create a new audio classification model, following the simple audio
# recognition tutorial by TensorFlow (https://www.tensorflow.org/tutorials/audio/simple_audio)


# To classify audio signals into different classes, you can use a deep learning approach 
# with a Convolutional Neural Network (CNN) or a Recurrent Neural Network (RNN). 
# Below is a simple example using a CNN with the TensorFlow and Keras library. 
# Make sure to install the required libraries if you haven't already

## ================================================================================



## 1) Project description
#-------------------------------------------








## 2) load modules
#-------------------------------------------
import os
import pathlib
import glob

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import models
from IPython import display
from scipy.signal import butter, lfilter

# Set the seed value for experiment reproducibility.
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)


# for now we will adapt the model so that we can test it on Orca and Minke whales.
# The original dataset on Speech recognition consists of over 105,000 audio files 
# in the WAV (Waveform) audio file format of people saying 35 different words. 
# This data was collected by Google and released under a CC BY license.

# => This means that we can eventually use 12 different whale calls (12 whale species 
#    found on Orkney) instead of 35 words. However, each species can have different
#    calls, which will be necessary to consider in the modelling process.


# Download and extract the mini_speech_commands.zip file containing the smaller 
# Speech Commands datasets with tf.keras.utils.get_file:
DATASET_PATH = 'data/mini_speech_commands'

data_dir = pathlib.Path(DATASET_PATH)
if not data_dir.exists():
  tf.keras.utils.get_file(
      'mini_speech_commands.zip',
      origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
      extract=True,
      cache_dir='.', cache_subdir='data')
  
# The dataset's audio clips are stored in eight folders corresponding to each speech 
# command: no, yes, down, go, left, up, right, and stop:
commands = np.array(tf.io.gfile.listdir(str(data_dir)))
commands = commands[(commands != 'README.md') & (commands != '.DS_Store')]
print('Commands:', commands)


## 3) Now we need to adapt the code from the tutorial to our project, identifying 
#     marine mammal species underwater by their calls
# --------------------------------------------------
DATASET_PATH = os.path.join('Documents', 'EMEC', 'Acoustics', 'Data_Acoustics', 
                            'SpeciesCalls')
data_dir = pathlib.Path(DATASET_PATH)

calls = np.array(['Orca', 'Minke'])
calls = calls[(calls != 'README.md') & (calls != '.DS_Store')]
print('Calls:', calls)

# before we create the audio classifier model, we want to apply a filter that removes
# background noises from the species calls. it is important to know the frequencies
# at which marine mammals communicate. According to research, mammal species around
# Orkney communicate within the following frequencies:
    
    # Orca: 600 Hz - 29000 Hz
    # Minke: 50 Hz - 9400 Hz
    


# define the filter functions (based on this post: https://stackoverflow.com/questions/12093594/how-to-implement-band-pass-butterworth-filter-with-scipy-signal-butter/12233959#12233959)
def butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype='band')

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


# use files and dirs


# loop through each species
for spp in calls:
  spp_dir = os.path.join(DATASET_PATH, spp)
  # loop through each subdir containing single calls
  # avoid hidden files (e.g., the .DSstore)
  for root, dirs, files in os.walk(spp_dir):
    print(root)
    files = [f for f in files if not f[0] == '.']
    dirs[:] = [d for d in dirs if not d[0] == '.']


    # Divided into directories this way, you can easily load the data 
    # using keras.utils.audio_dataset_from_directory.
    train_ds, val_ds = tf.keras.utils.audio_dataset_from_directory(
        directory=spp_dir,
        #labels=[],
        batch_size=64,
        validation_split=0.2,
        seed=42,
        output_sequence_length=16000,
        subset='both')

    label_names = np.array(train_ds.class_names)
    print()
    print("label names:", label_names)




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

# the code below is from Mike Polinowksi (see link above)



# import modules to jupyter notebook
import os
from matplotlib import pyplot as plt
import tensorflow as tf
import tensorflow_io as tfio



# Data Loading
# ========================

# To work with the recorded audio files we can select the corresponding data paths - in the example below 
# we pick both an example file that contains the signal we are looking for and one that only comes with background noise:
HARBORPORP_FILE = os.path.join('Documents', 'EMEC', 'Acoustics', 'Data_Acoustics', 'harborPorpoise.wav')
NOT_HARBORPORP_FILE = os.path.join('Documents', 'EMEC', 'Acoustics', 'Data_Acoustics', '3channel_sample_data_fromEMEC', 'channelABC_2023-07-25_14-09-33.wav')

# And read a single audio stream (mono) resampled to 16kHz from those files by feeding the filepath into the following function:
    # - decode_wav : Read single channel from stereo file
    # - squeeze : Since all the data is single channel (mono), drop the channels axis from array
    # - resample : Reduce audio data to 16kHz


def load_wav_16k_mono(filename):
    # Load encoded wav file
    file_contents = tf.io.read_file(filename)
    # Decode wav (tensors by channels) 
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
    # Removes trailing axis
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    # Goes from 44100Hz to 16000hz - amplitude of the audio signal
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    return wav


# We can run the function over both the positive and negative file:
wave = load_wav_16k_mono(HARBORPORP_FILE)
nwave = load_wav_16k_mono(NOT_HARBORPORP_FILE)


file_contents = tf.io.read_file(HARBORPORP_FILE)
# Decode wav (tensors by channels) 
wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
# Removes trailing axis
wav = tf.squeeze(wav, axis=-1)
sample_rate = tf.cast(sample_rate, dtype=tf.int64)
# Goes from 44100Hz to 16000hz - amplitude of the audio signal
wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
return wav
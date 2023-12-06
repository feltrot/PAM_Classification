
## initial code to create a audio classification model with 4 different classes

# Note: these may not be the most efficient models, but are a start
# ========================================================================

# To classify audio signals into different classes, you can use a deep learning approach 
# with a Convolutional Neural Network (CNN) or a Recurrent Neural Network (RNN). 
# Below is a simple example using a CNN with the TensorFlow and Keras library. 
# Make sure to install the required libraries if you haven't already:



## 1) model with one training dataset that is split into training and testing datsets
# -------------------------------------------------------------------------

## this model uses 1 entire dataset for audio signal classification
import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

# Function to extract features from audio files using librosa
def extract_features(file_path):
    audio, _ = librosa.load(file_path, duration=2.5, sr=16000, 
                            offset=0.5)#, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=13)
    return mfccs

# Function to load and preprocess the dataset
def load_data(data_path):
    labels = []
    features = []

    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                label = int(file.split('-')[1])  # Assuming file names are like "class-label.wav"
                features.append(extract_features(file_path))
                labels.append(label)

    return np.array(features), np.array(labels)

# Load the dataset
data_path = '/path/to/your/dataset'
features, labels = load_data(data_path)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, 
                                                    random_state=42)

# Define the CNN model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(X_train.shape[1], 
                                                                    X_train.shape[2], 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))  # Assuming 10 classes

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Reshape input data to fit the model
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2], 1))

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc}')

# Save the model
model.save('audio_classifier_model.h5')


# Make sure to replace '/path/to/your/dataset' with the path to your dataset. 
# This example uses MFCCs as features and a simple CNN architecture. Adjust the model 
# architecture and hyperparameters based on your specific requirements and dataset 
# characteristics.







#################################################

## 2) model where each class has its own dataset that are used as training
# to classify signals in a year audio signals 
# -------------------------------------------------------------------------
import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, models, utils

# Function to extract features from audio file
def extract_features(file_path):
    try:
        audio_data, _ = librosa.load(file_path)#, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio_data, sr=16000, n_mfcc=13)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print("Error encountered while parsing file: ", file_path)
        return None

# Function to load data from a directory
def load_data(directory):
    features = []
    labels = []
    for filename in os.listdir(directory):
        if filename.endswith(".wav"):
            file_path = os.path.join(directory, filename)
            class_label = directory.split('/')[-1]  # Assuming the class 
                                                    #label is the last part of 
                                                    # the directory path
            data = extract_features(file_path)
            if data is not None:
                features.append(data)
                labels.append(class_label)
    return features, labels

# Directory paths for your datasets
class1_HP = os.path.join('Documents', 'EMEC', 'Acoustics', 'Data_Acoustics', 
                         'HarbourPorpoise')
class2_PWS = os.path.join('Documents', 'EMEC', 'Acoustics', 'Data_Acoustics', 
                          'pacificWhiteSided')
class3_Minke = os.path.join('Documents', 'EMEC', 'Acoustics', 'Data_Acoustics',
                            'Minke')
class4_Rissos = os.path.join('Documents', 'EMEC', 'Acoustics', 'Data_Acoustics', 
                             'RissosDolphin')

# Load data for each class
class1_features, class1_labels = load_data(class1_HP)
class2_features, class2_labels = load_data(class2_PWS)
class3_features, class3_labels = load_data(class3_Minke)
class4_features, class4_labels = load_data(class4_Rissos)

# Combine features and labels from all classes
all_features = np.vstack([class1_features, class2_features, class3_features, 
                          class4_features])
all_labels = np.hstack([class1_labels, class2_labels, class3_labels, 
                        class4_labels])

# Encode labels to numerical values
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(all_labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(all_features, 
                                                    encoded_labels, 
                                                    test_size=0.2, 
                                                    random_state=42)

# Define the neural network model
model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(4, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc}')

# Save the model for future use
model.save('audio_classifier_model.h5')




# now use the model to predict marine life in the hydrophone recordings
# ======================================================================

# load the data
hydroPh_records = os.path.join('Documents', 'EMEC', 'Acoustics', 'Data_Acoustics', 
                          '3channel_sample_data_fromEMEC')

hydroPh_data = load_data(hydroPh_records)

predictions = model.predict(hydroPh_data)

# Make sure to replace 'path/to/class1_dataset', 'path/to/class2_dataset', 'path/to/class3_dataset', 
# and 'path/to/class4_dataset' with the actual paths to your datasets. 
# This example assumes that each class has its own directory containing the 
# corresponding .wav files. The code uses the MFCC (Mel-Frequency Cepstral Coefficients) 
# as features for audio classification.
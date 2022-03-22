import spaudiopy as spy
import csv
import math
import librosa
import numpy as np
import scipy.io.wavfile
import matplotlib.pyplot as plt
import copy
import tensorflow as tf

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

print(tf.__version__)

N = 4

# generating matrix with spherical harmonics    
micangles = []
with open('../MADA/mic_arr_shapes/eigenmike_theta_phi_rho.csv', newline='') as csvfile:
    csvfile = csv.reader(csvfile, delimiter=',')
    for row in csvfile:
        micangles.append([float(i)*math.pi/180 for i in row[:-1]])

micangles=list(map(list,zip(*micangles)))

A = spy.sph.sh_matrix(N, micangles[1], micangles[0], SH_type='real', weights=None)

# loading raw audio
x, fs = librosa.load('data/Beach-01-Raw.wav', sr=48000, mono=False)

nchans = 32
datapoints = []
targets = []
nsamps = 2
for isamp in np.random.choice(x.shape[1], size=nsamps, replace=False):
    for targetchan in range(nchans):
    
        Ax = A*x[:,isamp][:,np.newaxis]
        #Ax[targetchan] = 0
        Ax[targetchan] = A[targetchan]
        #Ax = Ax-A[targetchan][:,np.newaxis].T
        datapoints.append(Ax.flatten())
        #Ax = Ax-A[targetchan][:,np.newaxis].T
        #datapoints.append(Ax[np.arange(len(Ax))!=targetchan].flatten())
        targets.append(np.atleast_1d(x[targetchan,isamp]))

X = np.array(datapoints)
Y = np.array(targets)

train_features = X.copy()
train_labels = Y
normalizer = preprocessing.Normalization()
normalizer.adapt(train_features)

linear_model = tf.keras.Sequential([
    normalizer,
    layers.Dense(units=1, kernel_regularizer=tf.keras.regularizers.l1(0.01))
])

linear_model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.01),
    loss='mse')

history = linear_model.fit(
    train_features, train_labels, 
    epochs=10000,
    # suppress logging
    verbose=0,
    # Calculate validation results on 20% of the training data
    validation_split = 0.2)
print([loss for iloss, loss in enumerate(history.history['loss']) if iloss%100==0])

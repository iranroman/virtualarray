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
x, fs = librosa.load('data/Beach-01-Raw-8k.wav', sr=8000, mono=False)

nchans = 32
datapoints = []
targets = []
nsamps = 2048
ntime = 20
for isamp in np.random.choice(x.shape[1], size=nsamps, replace=False):
    for targetchan in range(nchans):
    
        Ax = [A*x[:,isamp+i-ntime+1][:,np.newaxis] for i in range(ntime)]
        for isamp in range(ntime):
            Ax[isamp][targetchan] = A[targetchan]
        datapoints.append(Ax)
        targets.append(np.atleast_1d(x[targetchan,isamp]))

X = np.array(datapoints).transpose(0,3,2,1)
Y = np.array(targets)

with open('data.npy', 'wb') as f:
	np.save(f, X)
	np.save(f, Y)

train_features = X.copy()
train_labels = Y
normalizer = preprocessing.Normalization(axis=[1,2,3])
normalizer.adapt(train_features)

model = tf.keras.models.Sequential()
model.add(normalizer)
model.add(layers.Conv2D(32, (4, 5), activation='relu', input_shape=(25, 32, 20), padding='valid'))
model.add(layers.MaxPooling2D((2, 2), padding='same'))
model.add(layers.Conv2D(64, (4, 5), activation='relu', padding='valid'))
model.add(layers.MaxPooling2D((2, 2), padding='same'))
model.add(layers.Conv2D(64, (4, 3), activation='relu', padding='valid'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1))

model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.001),
    loss='mse')

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
early_stopping = EarlyStopping(patience=1000)
checkpoint_path = 'model_checkpoints/'
checkpoint = ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
	monitor='val_loss',
    save_freq='epoch',
    verbose=1,
	save_best_only=True
)

history = model.fit(
    train_features, train_labels, 
    epochs=100000,
    # suppress logging
    verbose=1,
    # Calculate validation results on 20% of the training data
    validation_split = 0.2,
    callbacks = [early_stopping, checkpoint])

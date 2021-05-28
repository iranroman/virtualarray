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

# loading training data
with open('data.npy', 'rb') as f:
	X = np.load(f)
	Y = np.load(f)
train_features = X.copy()
train_labels = Y

# loading the model weights and restoring the model
model = tf.keras.models.Sequential()
model.add(preprocessing.Normalization(axis=[1,2,3], input_shape=(25,32,20)))
model.add(layers.Conv2D(32, (4, 5), activation='relu', input_shape=(25, 32, 20), padding='valid'))
model.add(layers.MaxPooling2D((2, 2), padding='same'))
model.add(layers.Conv2D(64, (4, 5), activation='relu', padding='valid'))
model.add(layers.MaxPooling2D((2, 2), padding='same'))
model.add(layers.Conv2D(64, (4, 3), activation='relu', padding='valid'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1))
model.load_weights('model_checkpoints/')
model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.001),
    loss='mse')
print(model.summary())
loss = model.evaluate(train_features,train_labels)
print(loss)

plt.imshow(train_features[14,:,:,-1])
plt.savefig('input.png')


# loading raw audio
x, fs = librosa.load('data/Beach-01-Raw-8k.wav', sr=8000, mono=False)

N = 4
# generating matrix with spherical harmonics to try with other samples 
micangles = []
with open('../MADA/mic_arr_shapes/eigenmike_theta_phi_rho.csv', newline='') as csvfile:
    csvfile = csv.reader(csvfile, delimiter=',')
    for row in csvfile:
        micangles.append([float(i)*math.pi/180 for i in row[:-1]])
micangles=list(map(list,zip(*micangles)))
A = spy.sph.sh_matrix(N, micangles[1], micangles[0], SH_type='real', weights=None)

nchans = 32
datapoints = []
targets = []
nsamps = 2
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
train_features = X.copy()
train_labels = Y

loss = model.evaluate(X,Y)
print(loss)

layer_outputs = [layer.output for layer in model.layers] 
activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(train_features[:32]) 
nchans=32
for ilayer in [1,3,5]:
    plt.imshow(np.mean(activations[ilayer][14,:,:,:],axis=-1))
    plt.savefig('14_'+str(ilayer)+'.png')

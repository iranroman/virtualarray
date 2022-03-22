import spaudiopy as spy
import csv
import math
import librosa
import numpy as np
import scipy.io.wavfile
import matplotlib.pyplot as plt
import copy
import tensorflow as tf
import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

print(tf.__version__)

N = 4

neighbors = {
    1 :[2 ,3 ,4 ,5 ,6 ,12],
    2 :[1 ,3 ,6 ,7 ,8],
    3 :[1 ,2 ,4 ,8 ,9 ,10],
    4 :[1 ,3 ,10,11,12],
    5 :[1 ,6 ,12,13,29],
    6 :[1 ,2 ,5 ,7 ,13,14],
    7 :[2 ,6 ,8 ,14,15,27],
    8 :[2 ,3 ,7 ,9 ,15,16],
    9 :[3 ,8 ,10,16,32],
    10:[3 ,4 ,9 ,11,31,32],
    11:[4 ,10,12,23,30,31],
    12:[1 ,4 ,5 ,11,29,30],
    13:[5 ,6 ,14,21,28,29],
    14:[6 ,7 ,13,27,28],
    15:[7 ,8 ,16,26,27],
    16:[8 ,9 ,15,25,26],
    17:[18,19,20,21,22,28],
    18:[17,19,22,23,24],
    19:[17,18,20,24,25,26],
    20:[17,19,26,27,28],
    21:[13,17,22,28,29],
    22:[17,18,21,23,29,30],
    23:[18,22,30,31,11],
    24:[18,19,23,25,31,32],
    25:[16,19,24,26,32],
    26:[15,16,19,20,25,27],
    27:[14,15,20,26,28,7],
    28:[13,14,17,20,21,27],
    29:[5 ,12,13,21,22,30],
    30:[11,12,29,22,23,29],
    31:[10,11,32,23,24],
    32:[9 ,10,31,24,25],
}

# generating matrix with spherical harmonics    
micangles = []
with open('../MADA/mic_arr_shapes/eigenmike_theta_phi_rho.csv', newline='') as csvfile:
    csvfile = csv.reader(csvfile, delimiter=',')
    for row in csvfile:
        micangles.append([float(i)*math.pi/180 for i in row[:-1]])

micangles=list(map(list,zip(*micangles)))

A = spy.sph.sh_matrix(N, micangles[1], micangles[0], SH_type='real', weights=None).T

files = os.listdir('../../iranroman/datasets/aggregate/')

X = []
Y = []
for f in files:
# loading raw audio
    print(f)
    x, fs = librosa.load('../../iranroman/datasets/aggregate/'+f, sr=8000, mono=False)

    nchans = 32
    datapoints = []
    targets = []
    nsamps = 64
    ntime = 20
    for isamp in np.random.choice(int(x.shape[1]*0.8), size=nsamps, replace=False):
        for targetchan in range(nchans):
        
            Ax = [x[:,isamp+i-ntime+1][:,np.newaxis] for i in range(ntime)]
            for isamp in range(ntime):
                Ax[isamp][targetchan] = 1
                for neighchan in neighbors[targetchan+1]:
                    Ax[isamp][neighchan-1] = 1
            datapoints.append(Ax)
            targets.append(np.atleast_1d(x[targetchan,isamp]))

    X.append(np.array(datapoints).transpose(0,3,2,1))
    Y.append(np.array(targets))

X = np.vstack(X)
Y = np.vstack(Y)

shuffle_samples = np.random.choice(X.shape[0], size=X.shape[0], replace=False)
X = X[shuffle_samples]
Y = Y[shuffle_samples]


with open('data.npy', 'wb') as f:
	np.save(f, X)
	np.save(f, Y)

train_features = X.copy()
train_labels = Y

input_data = tf.keras.Input(shape=(1,32,20))
sph_harm = tf.keras.Input(shape=(25,32,20))
norm = preprocessing.Normalization(axis=[2,3])(input_data)
rep = layers.Lambda(lambda x: tf.keras.backend.repeat_elements(x=x, rep=25, axis=1))(norm)
mult = layers.Multiply()([sph_harm,rep])
conv1 = layers.Conv2D(32, (4, 5), activation='relu', input_shape=(25, 32, 20), padding='valid')(mult)
maxp1 = layers.MaxPooling2D((2, 2), padding='same')(conv1)
conv2 = layers.Conv2D(64, (4, 5), activation='relu', padding='valid')(maxp1)
maxp2 = layers.MaxPooling2D((2, 2), padding='same')(conv2)
conv3 = layers.Conv2D(64, (4, 3), activation='relu', padding='valid')(maxp2)
flat = layers.Flatten()(conv3)
dense = layers.Dense(64, activation='relu')(flat)
out  = layers.Dense(1)(dense)

model = tf.keras.Model(inputs=[input_data,sph_harm], outputs=out)
model.summary()

#normalizer = preprocessing.Normalization(axis=[1,2,3])
#normalizer.adapt(train_features)
#model = tf.keras.models.Sequential()
#model.add(normalizer)
#model.add(layers.RepeatVector(25))
#model.add(layers.MultiplyRepeatVector(25))
#model.add(layers.Conv2D(32, (4, 5), activation='relu', input_shape=(25, 32, 20), padding='valid'))
#model.add(layers.MaxPooling2D((2, 2), padding='same'))
#model.add(layers.Conv2D(64, (4, 5), activation='relu', padding='valid'))
#model.add(layers.MaxPooling2D((2, 2), padding='same'))
#model.add(layers.Conv2D(64, (4, 3), activation='relu', padding='valid'))
#model.add(layers.Flatten())
#model.add(layers.Dense(64, activation='relu'))
#model.add(layers.Dense(1))

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

ntest = 10000
test_samples = np.random.choice(train_features.shape[0], size=ntest, replace=False)
loss = model.evaluate(
            [train_features[test_samples], np.tile(A[np.newaxis,:,:,np.newaxis],(ntest,1,1,20))],
            train_labels[test_samples],
            batch_size = 8192
        )
print(loss)

history = model.fit(
    [train_features, np.tile(A[np.newaxis,:,:,np.newaxis],(train_features.shape[0],1,1,20))], train_labels, 
    epochs=100000,
    # suppress logging
    verbose=1,
    # Calculate validation results on 20% of the training data
    validation_split = 0.2,
    batch_size=8192,
    callbacks = [early_stopping, checkpoint])

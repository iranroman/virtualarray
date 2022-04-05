import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
import dynfilt_layers


def unet(pretrained_weights = None, input_size = (256,256,1), coords_size = (3,3), drop=0.5):

    inputc = Input(coords_size)
    ctr1 = Dense(512, activation="relu")(inputc)
    ctr2 = Dense(33*64)(ctr1)
    ctr2 = Reshape(1,33,3,64)(ctr2)

    inputs = Input(input_size)
    conv1 = dynfilt_layers.Conv2D(padding="SAME")(inputs, inputsc)
    conv1 = Conv2D(64, (1,33), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(1, 2))(conv1)
    conv2 = Conv2D(128, (1,33), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, (1,33), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(1, 2))(conv2)
    conv3 = Conv2D(256, (1,33), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, (1,33), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(1, 2))(conv3)
    conv4 = Conv2D(512, (1,33), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, (1,33), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(drop)(conv4)
    pool4 = MaxPooling2D(pool_size=(1, 2))(drop4)

    conv5 = Conv2D(1024, (1,33), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, (1,33), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(drop)(conv5)

    up6 = Conv2D(512, (1,33), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (1,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, (1,33), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, (1,33), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, (1,33), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (1,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, (1,33), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, (1,33), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, (1,33), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (1,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, (1,33), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, (1,33), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, (1,33), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (1,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, (1,33), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, (1,33), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(1, (1,33), activation = 'tanh', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    #conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(inputs = (inputs, inputc), outputs = conv9)

    model.compile(optimizer = adam_v2.Adam(lr = 1e-5), loss = 'mse', metrics = ['MeanSquaredError'])
    
    #model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model



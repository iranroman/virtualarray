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


def unet(pretrained_weights = None,input_size = (256,256,1), drop=0.5):
    inputs = Input(input_size)
    out = Lambda(lambda x: tf.expand_dims(tf.reduce_mean(x,axis=-1), -1))(inputs)

    model = Model(inputs = inputs, outputs = out)

    model.compile(optimizer = adam_v2.Adam(lr = 1e-6), loss = 'mse', metrics = ['MeanSquaredError'])
    
    return model



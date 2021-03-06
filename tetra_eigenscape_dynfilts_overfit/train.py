import numpy as np
from model import unet
import pickle
import os
from data_gen import DataGenerator
import tensorflow as tf
from micarraylib.arraycoords.core import micarray
from micarraylib.arraycoords import array_shapes_raw

eigenmike = micarray(array_shapes_raw.eigenmike_raw, "polar", "degrees", "Eigenmike")
eigenmike.standard_coords('cartesian')

with open('tetras_dict.pkl', 'rb') as f:
    tetras_dict = pickle.load(f)


vl_filenames = [
    'ShoppingCentre-06-Raw',
]

ts_filenames = [
    'Beach-03-Raw',
    'BusyStreet-06-Raw',
    'PedestrianZone-02-Raw',
    'QuietStreet-01-Raw',
    'ShoppingCentre-05-Raw',
    'TrainStation-08-Raw',
    'Woodland-03-Raw',
]


data_files = [f for f in os.listdir('/scratch/irr2020/unet/data_npy/eigenscape/') if f.endswith(".npy")]

tr_filenames = sorted([f[:-4] for f in data_files if f not in vl_filenames and f not in ts_filenames])

training_generator = DataGenerator(vl_filenames, dim=(1,1920), n_channels=3, tetras_dict=tetras_dict, array_coords_dict=eigenmike.coords_dict)
validation_generator = DataGenerator(vl_filenames, dim=(1,1920), n_channels=3, tetras_dict=tetras_dict, array_coords_dict=eigenmike.coords_dict)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath='./chk/checkpoint',
                                                save_weights_only=True,
                                                verbose=1,
                                                save_best_only=True,
                                                monitor='val_loss',
                                                mode='min'
                                                )


model = unet(input_size=(1,1920,3))
model.summary()
    
model.fit(x=training_generator,validation_data=validation_generator,epochs=1000,callbacks=[cp_callback], shuffle=True, verbose=2)

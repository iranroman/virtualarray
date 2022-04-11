import numpy as np
from model import unet
import pickle
import os
from data_gen import DataGenerator
import tensorflow as tf


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


model = unet(input_size=(1,1920,3))
model.summary()
model.load_weights('./chk/checkpoint')

for i in range(len(vl_filenames)):
    print("prediction with subset",vl_filenames[i])
    validation_generator = DataGenerator([vl_filenames[i]], dim=(1,1920), n_channels=3, tetras_dict=tetras_dict)
    out = model.predict(validation_generator, verbose=1)
    np.save('./predictions/{}'.format(vl_filenames[i]),out)
    print(' ')

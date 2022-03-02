import librosa
import librosa.display
import numpy as np
import tensorflow as tf
from itertools import chain, combinations
from micarraylib.datasets import marco, eigenscape_raw
from micarraylib.arraycoords.array_shapes_utils import _polar2cart
import matplotlib.pyplot as plt
import sys
from copy import deepcopy
from sklearn.neighbors import NearestNeighbors


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


marco_dataset = marco(download=False, data_home='/home/iran/datasets/marco')
#marco_dataset = eigenscape_raw(download=False, data_home='/home/iran/datasets/eigenscape_raw')
marco_dataset.fs = 16000
fs = marco_dataset.fs
datapoint_dur = 0.1
lr = 1e-6
patience = 10
edge = 0

def extract_features(X, fs, datapoint_dur, nfft = 4096, hop_length=512):

    X = np.array([[[x[i:i+int(datapoint_dur*fs)]] for i in range(0,len(x)-5*int(datapoint_dur*fs),int(datapoint_dur*fs))] for x in X]).transpose(1,2,3,0)
    X = X/(np.max(X[:,edge:,edge:],axis=tuple(range(1,X.ndim)),keepdims=True)+1e-30)
    return X

mse = tf.losses.MeanSquaredError()

# training and evaluation routine
marco_data = {m:{} for m in marco_dataset.array_names}
for array in marco_data.keys():
    for ss in ['piano_solo_2']:
        print('getting ', array, ' ', ss)
        X = marco_dataset.get_audio_numpy(ss,array)

        # extract features
        marco_data[array][ss] = extract_features(X, fs, datapoint_dur)
epoch_error = 0
for array in marco_data.keys():

    coords, capsule_names = marco_dataset.get_capsule_coords_numpy(array)
    coords = np.array([v for v in _polar2cart({j:i for i,j in zip(coords,capsule_names)}, units='radians').values()])

    for ss in marco_data[array].keys():

        data = marco_data[array][ss]
        nchans = data.shape[-1]

        total_error = 0
        for ichan in range(nchans):

            batch_size = 512
            batch_idx = np.random.choice(data.shape[0],batch_size,replace=False)
            Y = data[batch_idx][:,edge:,edge:,[ichan]]
            out = np.zeros_like(Y)
            error = mse(out,Y)
            total_error += error 
        fig, axs = plt.subplots(3, 1, figsize=(10, 5))
        axs[0].plot(np.squeeze(Y[0]))
        axs[0].axis(ymin=-1.0,ymax=1.0)
        axs[0].grid(True)
        axs[1].plot(np.squeeze(out[0]))
        axs[1].axis(ymin=-1.0,ymax=1.0)
        axs[1].grid(True)
        axs[2].plot(np.abs(np.squeeze(out[0])-np.squeeze(Y[0])))
        axs[2].axis(ymin=-1.0,ymax=1.0)
        axs[2].grid(True)
        plt.savefig('_'.join(['plots/eval_inference_zeros',ss,array,'.png']))
        plt.close()
        Xerr = np.mean(np.squeeze(out-Y),axis=0)
        plt.figure()
        librosa.display.specshow(librosa.amplitude_to_db(np.abs(librosa.stft(Xerr, n_fft=256)), ref=np.max),sr=fs,hop_length=64,x_axis='time', y_axis='linear')
        plt.colorbar()
        plt.savefig('_'.join(['plots/eval_inference_zeros_fft_error',ss,array,'.png']))
        plt.close()

    print('-----------------------------')
    print('                             ')
    print('For microphone', array,', the average error was', total_error/nchans)
    print('                             ')
    print('-----------------------------')

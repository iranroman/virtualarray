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

# Helper functions to calculate angle between capsules
# from https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)
def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


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

    source_error = 0
    for ss in marco_data[array].keys():

        data = marco_data[array][ss]
        nchans = data.shape[-1]

        for in_size in [1]:

            total_error = 0
            distances = []
            angles = []
            for ichan in range(nchans):

                target_chan = deepcopy(np.array(coords[ichan]))
                coords[ichan] = np.array([1000,1000,1000])
                nbrs = NearestNeighbors(n_neighbors=1).fit(coords)
                _, all_chans_in = nbrs.kneighbors([target_chan])
                ncombs = len(all_chans_in)
                coords[ichan] = target_chan

                comb_error = 0
                for chans_in in all_chans_in:

                    batch_size = 512
                    batch_idx = np.random.choice(data.shape[0],batch_size,replace=False)
                    X = data[batch_idx][...,np.array(chans_in)]
                    Y = data[batch_idx][:,edge:,edge:,[ichan]]
                    Xc = coords[np.array(chans_in)].astype(np.float32)[0]
                    Yc = coords[[ichan]].astype(np.float32)[0]
                    distances.append(np.linalg.norm(Xc-Yc))
                    angles.append(angle_between(Xc, Yc))
                    out = np.mean(X,axis=-1,keepdims=True)
                    error = mse(out,Y)
                    print(error)
                    comb_error += error.numpy()
                total_error += comb_error/ncombs 
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
            plt.savefig('_'.join(['plots/eval_inference_1NN',ss,array,str(in_size),'.png']))
            plt.close()
            Xerr = np.mean(np.squeeze(out-Y),axis=0)
            plt.figure()
            librosa.display.specshow(librosa.amplitude_to_db(np.abs(librosa.stft(Xerr, n_fft=256)), ref=np.max),sr=fs,hop_length=64,x_axis='time', y_axis='linear')
            plt.colorbar()
            plt.savefig('_'.join(['plots/eval_inference_1NN_fft_error',ss,array,str(in_size),'.png']))
            plt.close()
    print('-----------------------------')
    print('                             ')
    print('For array', array,', thea average error was', total_error/nchans)
    print('With mean angle of', 180*np.mean(angles)/np.pi, ' degrees, and mean distance of', np.mean(distances))
    print('                             ')
    print('-----------------------------')

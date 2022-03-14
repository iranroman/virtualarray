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
import itertools
import operator

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

marco_dataset = marco(download=False, data_home='/home/iran/datasets/marco')
marco_dataset.fs = 16000
fs = marco_dataset.fs
datapoint_dur = 0.1
lr = 1e-6
patience = 10
edge = 0


rings = {
    '1':[np.array([5,6,2,3,4,12]),np.array([13,14,7,8,9,10,11,30,29]),np.array([21,28,27,15,16,32,31,23,22]),np.array([17,20,26,25,24,18]),np.array([19])],
    '2':[np.array([6,7,8,3,1]),np.array([5,14,15,9,4]),np.array([13,27,16,10,12]),np.array([29,28,26,32,11]),np.array([21,20,25,31,30]),np.array([17,19,24,23,22]),np.array([18])],
}


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
marco_data = {"Eigenmike":{}}#m:{} for m in marco_dataset.array_names}
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


        total_error = 0
        distances = []
        angles = []
        errors = []
        for ichan in range(0,2):

            numElems = 4
            all_chans_in = [l[np.round(np.linspace(0, l.shape[0] - 1, numElems)).astype(int)]-1 for l in rings[str(ichan+1)]]
            ncombs = len(all_chans_in)

            comb_error = 0
            for chans_in in all_chans_in:

                batch_size = 512
                batch_idx = np.random.choice(data.shape[0],batch_size,replace=False)
                X = data[batch_idx][...,np.array(chans_in)]
                Y = data[batch_idx][:,edge:,edge:,[ichan]]
                Xc = coords[np.array(chans_in)].astype(np.float32)
                Yc = coords[[ichan]].astype(np.float32)[0]
                distance = np.mean([np.linalg.norm(Xc[3]-Yc), np.linalg.norm(Xc[2]-Yc), np.linalg.norm(Xc[0]-Yc), np.linalg.norm(Xc[1]-Yc)])
                angle = np.mean([angle_between(Xc[3], Yc),angle_between(Xc[2], Yc),angle_between(Xc[0], Yc),angle_between(Xc[1],Yc)])
                out = np.mean(X,axis=-1,keepdims=True)
                error = mse(out,Y)
                errors.append(error)
                angles.append(round(180*angle/np.pi,-1))
                distances.append(round(distance,2))
        errors = np.array(errors)
        angles = np.array(angles)
        distances = np.array(distances)
        ang_idx = np.argsort(angles)
        angles = angles[ang_idx]
        errors_angles = errors[ang_idx]
        dis_idx = np.argsort(distances)
        distances = distances[dis_idx]
        errors_distances = errors[dis_idx]
        angle_bins = [[e[0] for e in d[1]] for d in itertools.groupby(enumerate(angles), key=operator.itemgetter(1))]
        distance_bins = [[e[0] for e in d[1]] for d in itertools.groupby(enumerate(distances), key=operator.itemgetter(1))]
        plt.figure(figsize=(6,6))
        for b in angle_bins:
            plt.yscale('log')
            plt.boxplot(errors_angles[b],positions=[int(angles[b][0])],widths=[8])
            plt.xlabel('degrees')
            plt.xlim([30, 190])
        plt.grid()
        plt.savefig('_'.join(['plots/angles_vs_errors_rings_4NN',ss,array,'.png']))
        plt.close()
        plt.figure(figsize=(6,6))
        for b in distance_bins:
            plt.yscale('log')
            plt.boxplot(errors_distances[b],positions=[distances[b][0]],widths=[0.008])
            plt.xlabel('meters')
            plt.xlim([0.02, 0.09])
        plt.grid()
        plt.savefig('_'.join(['plots/distances_vs_errors_rings_4NN',ss,array,'.png']))
        plt.close()

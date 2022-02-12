import librosa
import librosa.display
import numpy as np
import tensorflow as tf
from itertools import chain, combinations
from micarraylib.datasets import marco
from micarraylib.arraycoords.array_shapes_utils import _polar2cart
import matplotlib.pyplot as plt
import sys
from copy import deepcopy
from sklearn.neighbors import NearestNeighbors


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

def extract_features(X, fs, datapoint_dur, nfft = 4096, hop_length=512):

    X = np.array([[[x[i:i+int(datapoint_dur*fs)]] for i in range(0,len(x)-5*int(datapoint_dur*fs),int(datapoint_dur*fs))] for x in X]).transpose(1,2,3,0)
    X = X/np.max(X[:,edge:,edge:],axis=tuple(range(1,X.ndim)),keepdims=True)
    return X

# CNN parameters
nchans = 64
K = 33
ch1 = nchans*1
ch2 = ch1
ch3 = nchans*2
ch4 = ch3
ch5 = nchans*4
ch6 = ch5
ch7 = nchans*8
ch8 = ch7
ch9 = nchans*16
ch10 = ch9
ch11 = ch8
ch12 = ch8
ch13 = ch8
ch14 = ch6
ch15 = ch6
ch16 = ch6
ch17 = ch4
ch18 = ch4
ch19 = ch4
ch20 = ch2
ch21 = ch2
ch22 = ch2
ch23 = 1

# capsule coordinates network parameters
sH1 = 256
sH2 = 512
sH3 = 1*K*(ch1)

oW1  = tf.constant(np.load('best_params/parameter_0.npy'))
ob1  = tf.constant(np.load('best_params/parameter_1.npy'))
oW2  = tf.constant(np.load('best_params/parameter_2.npy'))
ob2  = tf.constant(np.load('best_params/parameter_3.npy'))
oW3  = tf.constant(np.load('best_params/parameter_4.npy'))
ob3  = tf.constant(np.load('best_params/parameter_5.npy'))
sW1  = tf.constant(np.load('best_params/parameter_6.npy'))
sb1  = tf.constant(np.load('best_params/parameter_7.npy'))
sW2  = tf.constant(np.load('best_params/parameter_8.npy'))
sb2  = tf.constant(np.load('best_params/parameter_9.npy'))
sW3  = tf.constant(np.load('best_params/parameter_10.npy'))
sb3  = tf.constant(np.load('best_params/parameter_11.npy'))
F1   = tf.constant(np.load('best_params/parameter_12.npy'))
F2   = tf.constant(np.load('best_params/parameter_13.npy'))
F3   = tf.constant(np.load('best_params/parameter_14.npy'))
F4   = tf.constant(np.load('best_params/parameter_15.npy'))
F5   = tf.constant(np.load('best_params/parameter_16.npy'))
F6   = tf.constant(np.load('best_params/parameter_17.npy'))
F7   = tf.constant(np.load('best_params/parameter_18.npy'))
F8   = tf.constant(np.load('best_params/parameter_19.npy'))
F9   = tf.constant(np.load('best_params/parameter_20.npy'))
F10  = tf.constant(np.load('best_params/parameter_21.npy'))
F11  = tf.constant(np.load('best_params/parameter_22.npy'))
F12  = tf.constant(np.load('best_params/parameter_23.npy'))
F13  = tf.constant(np.load('best_params/parameter_24.npy'))
F14  = tf.constant(np.load('best_params/parameter_25.npy'))
F15  = tf.constant(np.load('best_params/parameter_26.npy'))
F16  = tf.constant(np.load('best_params/parameter_27.npy'))
F17  = tf.constant(np.load('best_params/parameter_28.npy'))
F18  = tf.constant(np.load('best_params/parameter_29.npy'))
F19  = tf.constant(np.load('best_params/parameter_30.npy'))
F20  = tf.constant(np.load('best_params/parameter_31.npy'))
F21  = tf.constant(np.load('best_params/parameter_32.npy'))
F22  = tf.constant(np.load('best_params/parameter_33.npy'))
F23  = tf.constant(np.load('best_params/parameter_34.npy'))
F11b = tf.constant(np.load('best_params/parameter_35.npy'))
F14b = tf.constant(np.load('best_params/parameter_36.npy'))
F17b = tf.constant(np.load('best_params/parameter_37.npy'))
F20b = tf.constant(np.load('best_params/parameter_38.npy'))
b1   = tf.constant(np.load('best_params/parameter_39.npy'))
b2   = tf.constant(np.load('best_params/parameter_40.npy'))
b3   = tf.constant(np.load('best_params/parameter_41.npy'))
b4   = tf.constant(np.load('best_params/parameter_42.npy'))
b5   = tf.constant(np.load('best_params/parameter_43.npy'))
b6   = tf.constant(np.load('best_params/parameter_44.npy'))
b7   = tf.constant(np.load('best_params/parameter_45.npy'))
b8   = tf.constant(np.load('best_params/parameter_46.npy'))
b9   = tf.constant(np.load('best_params/parameter_47.npy'))
b10  = tf.constant(np.load('best_params/parameter_48.npy'))
b11  = tf.constant(np.load('best_params/parameter_49.npy'))
b12  = tf.constant(np.load('best_params/parameter_50.npy'))
b13  = tf.constant(np.load('best_params/parameter_51.npy'))
b14  = tf.constant(np.load('best_params/parameter_52.npy'))
b15  = tf.constant(np.load('best_params/parameter_53.npy'))
b16  = tf.constant(np.load('best_params/parameter_54.npy'))
b17  = tf.constant(np.load('best_params/parameter_55.npy'))
b18  = tf.constant(np.load('best_params/parameter_56.npy'))
b19  = tf.constant(np.load('best_params/parameter_57.npy'))
b20  = tf.constant(np.load('best_params/parameter_58.npy'))
b21  = tf.constant(np.load('best_params/parameter_59.npy'))
b22  = tf.constant(np.load('best_params/parameter_60.npy'))
b23  = tf.constant(np.load('best_params/parameter_61.npy'))
b11b = tf.constant(np.load('best_params/parameter_62.npy'))
b14b = tf.constant(np.load('best_params/parameter_63.npy'))
b17b = tf.constant(np.load('best_params/parameter_64.npy'))
b20b = tf.constant(np.load('best_params/parameter_65.npy'))

# optimizer variables
mse = tf.losses.MeanSquaredError()
all_vars = [oW1,ob1,oW2,ob2,oW3,ob3,sW1,sb1,sW2,sb2,sW3,sb3,F1,F2,F3,F4,F5,F6,F7,F8,F9,F10,F11,F12,F13,F14,F15,F16,F17,F18,F19,F20,F21,F22,F23,F11b,F14b,F17b,F20b,b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11,b12,b13,b14,b15,b16,b17,b18,b19,b20,b21,b22,b23,b11b,b14b,b17b,b20b]

# The network architecture
@tf.function()
def forward_pass(X,Xc,Yc,all_vars,N,H,W,chans_in,ch1,droprate):

    # unpack variables
    oW1,ob1,oW2,ob2,oW3,ob3,sW1,sb1,sW2,sb2,sW3,sb3,F1,F2,F3,F4,F5,F6,F7,F8,F9,F10,F11,F12,F13,F14,F15,F16,F17,F18,F19,F20,F21,F22,F23,F11b,F14b,F17b,F20b,b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11,b12,b13,b14,b15,b16,b17,b18,b19,b20,b21,b22,b23,b11b,b14b,b17b,b20b = all_vars

    # process input capsule coordinates
    sh1 = tf.nn.relu(tf.add(tf.matmul(Xc, sW1), sb1))
    sh2 = tf.nn.relu(tf.add(tf.matmul(sh1, sW2), sb2))
    sh3 = tf.add(tf.matmul(sh2, sW3), sb3)
    F = tf.transpose(tf.reshape(sh3, (chans_in,1,K,ch1)), perm=(1,2,0,3))

    # process output chapsule coordinates
    sf1 = tf.nn.relu(tf.add(tf.matmul(Yc, oW1), ob1))
    sf2 = tf.nn.relu(tf.add(tf.matmul(sf1, oW2), ob2))
    Fo = tf.add(tf.matmul(sf2, oW3), ob3)
    Fo = tf.transpose(tf.reshape(Fo, (1,1,K,ch1)), perm=(1,2,3,0))
    
    # combine filters
    F1d = F + 0*tf.repeat(F1,chans_in,axis=2)
    F23d = Fo + 0*F23

    ################
    # forward pass #
    ################

    # first downsample
    out = tf.divide(tf.nn.conv2d(X,F1d,1,padding='SAME'),chans_in)
    out = tf.nn.relu(tf.nn.bias_add(out,b1))
    out = tf.nn.conv2d(out,F2,1,padding='SAME')
    out1 = tf.nn.relu(tf.nn.bias_add(out,b2))
    out = tf.nn.max_pool2d(out1, ksize=2, strides=(1,2),padding='SAME')
    out = tf.nn.dropout(out, droprate)

    # second downsample
    out = tf.nn.conv2d(out,F3,1,padding='SAME')
    out = tf.nn.relu(tf.nn.bias_add(out,b3))
    out = tf.nn.conv2d(out,F4,1,padding='SAME')
    out2 = tf.nn.relu(tf.nn.bias_add(out,b4))
    out = tf.nn.max_pool2d(out2, ksize=2, strides=(1,2),padding='SAME')
    out = tf.nn.dropout(out, droprate)

    # third downsample
    out = tf.nn.conv2d(out,F5,1,padding='SAME')
    out = tf.nn.relu(tf.nn.bias_add(out,b5))
    out = tf.nn.conv2d(out,F6,1,padding='SAME')
    out3 = tf.nn.relu(tf.nn.bias_add(out,b6))
    out = tf.nn.max_pool2d(out3, ksize=2, strides=(1,2),padding='SAME')
    out = tf.nn.dropout(out, droprate)

    # fourth downsample
    out = tf.nn.conv2d(out,F7,1,padding='SAME')
    out = tf.nn.relu(tf.nn.bias_add(out,b7))
    out = tf.nn.conv2d(out,F8,1,padding='SAME')
    out4 = tf.nn.relu(tf.nn.bias_add(out,b8))
    out = tf.nn.max_pool2d(out4, ksize=2, strides=(1,2),padding='SAME')
    out = tf.nn.dropout(out, droprate)

    # bottom
    out = tf.nn.conv2d(out,F9,1,padding='SAME')
    out = tf.nn.relu(tf.nn.bias_add(out,b9))
    out = tf.nn.conv2d(out,F10,1,padding='SAME')
    out = tf.nn.relu(tf.nn.bias_add(out,b10))

    # first upsample
    out = tf.nn.conv2d_transpose(out,F11, out4.shape, (1,2), padding="SAME")
    out = tf.nn.bias_add(out,b11)
    out = tf.concat([out,out4],-1)
    out = tf.nn.conv2d(out,F11b, 1, padding="VALID")
    out = tf.nn.bias_add(out,b11b)
    out = tf.nn.dropout(out, droprate)
    out = tf.nn.conv2d(out,F12,1,padding='SAME')
    out = tf.nn.relu(tf.nn.bias_add(out,b12))
    out = tf.nn.conv2d(out,F13,1,padding='SAME')
    out = tf.nn.relu(tf.nn.bias_add(out,b13))

    # second upsample
    out = tf.nn.conv2d_transpose(out,F14, out3.shape, (1,2), padding="SAME")
    out = tf.nn.bias_add(out,b14)
    out = tf.concat([out,out3],-1)
    out = tf.nn.conv2d(out,F14b, 1, padding="VALID")
    out = tf.nn.bias_add(out,b14b)
    out = tf.nn.dropout(out, droprate)
    out = tf.nn.conv2d(out,F15,1,padding='SAME')
    out = tf.nn.relu(tf.nn.bias_add(out,b15))
    out = tf.nn.conv2d(out,F16,1,padding='SAME')
    out = tf.nn.relu(tf.nn.bias_add(out,b16))

    # third upsample
    out = tf.nn.conv2d_transpose(out,F17, out2.shape, (1,2), padding="SAME")
    out = tf.nn.bias_add(out,b17)
    out = tf.concat([out,out2],-1)
    out = tf.nn.conv2d(out,F17b, 1, padding="VALID")
    out = tf.nn.bias_add(out,b17b)
    out = tf.nn.dropout(out, droprate)
    out = tf.nn.conv2d(out,F18,1,padding='SAME')
    out = tf.nn.relu(tf.nn.bias_add(out,b18))
    out = tf.nn.conv2d(out,F19,1,padding='SAME')
    out = tf.nn.relu(tf.nn.bias_add(out,b19))

    # fourth upsample
    out = tf.nn.conv2d_transpose(out,F20, out1.shape, (1,2), padding="SAME")
    out = tf.nn.bias_add(out,b20)
    out = tf.concat([out,out1],-1)
    out = tf.nn.conv2d(out,F20b, 1, padding="VALID")
    out = tf.nn.bias_add(out,b20b)
    out = tf.nn.dropout(out, droprate)
    out = tf.nn.conv2d(out,F21,1,padding='SAME')
    out = tf.nn.relu(tf.nn.bias_add(out,b21))
    out = tf.nn.conv2d(out,F22,1,padding='SAME')
    out = tf.nn.relu(tf.nn.bias_add(out,b22))

    # output
    out = tf.nn.conv2d(out, F23d, 1, padding='SAME')
    out = tf.nn.tanh(tf.nn.bias_add(out,b23))

    return out

# training and evaluation routine
print("====================================")
print("Evaluation after training")
print("====================================")
marco_data = {'Eigenmike':{}}#m:{} for m in marco_dataset.array_names}
for array in marco_data.keys():
    for ss in ['single_sources0deg']:
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

        in_size_error = 0
        for in_size in [3]:#reversed(range(nchans-1,nchans)):

            total_error = 0
            for ichan in range(nchans):

                #chan_range = list(range(nchans))
                #chan_range.remove(ichan)

                #all_chans_in = combinations(chan_range, in_size)
                #ncombs = sum(1 for i in all_chans_in)
                #all_chans_in = combinations(chan_range, in_size)
                target_chan = deepcopy(np.array(coords[ichan]))
                coords[ichan] = np.array([1000,1000,1000])
                nbrs = NearestNeighbors(n_neighbors=3).fit(coords)
                _, all_chans_in = nbrs.kneighbors([target_chan])
                ncombs = len(all_chans_in)
                coords[ichan] = target_chan
                
                comb_error = 0
                for chans_in in all_chans_in:

                    batch_size = 512
                    batch_idx = np.random.choice(data.shape[0],batch_size,replace=False)
                    X = data[batch_idx][...,np.array(chans_in)]
                    Xc = coords[np.array(chans_in)].astype(np.float32)
                    Y = data[batch_idx][:,edge:,edge:,[ichan]]
                    Yc = coords[[ichan]].astype(np.float32)
                    N, H, W, chans_in = X.shape
                    out = forward_pass(X,Xc,Yc,all_vars,N,H,W,chans_in,ch1,0)
                    error = mse(out,Y)
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
            plt.savefig('_'.join(['plots/eval_inference',ss,array,str(in_size),'.png']))
            plt.close()
            Xerr = np.squeeze(out[0])-np.squeeze(Y[0])
            plt.figure()
            librosa.display.specshow(librosa.amplitude_to_db(np.abs(librosa.stft(Xerr, n_fft=256)), ref=np.max),sr=fs,hop_length=64,x_axis='time', y_axis='linear')
            plt.colorbar()
            plt.savefig('_'.join(['plots/eval_inference_fft_error',ss,array,str(in_size),'.png']))
            plt.close()
            in_size_error += total_error/nchans 
            print(array,ss,'No. input chans',in_size, 'MSE:', total_error/nchans)
        source_error += in_size_error/len(range(nchans-1,nchans))

    epoch_error += source_error/len(marco_data[array].keys())
print('-----------------------------')
print('                             ')
print('Average error was', epoch_error/len(marco_data.keys()))
print('                             ')
print('-----------------------------')

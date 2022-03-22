import librosa
import numpy as np
import tensorflow as tf
from itertools import chain, combinations
from micarraylib.datasets import marco
from micarraylib.arraycoords.array_shapes_utils import _polar2cart
import matplotlib.pyplot as plt
import sys
from sklearn.neighbors import NearestNeighbors


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


marco_dataset = marco(download=False, data_home='/home/iran/datasets/marco')
marco_dataset.fs = 16000
fs = marco_dataset.fs
datapoint_dur = 0.1
lr = 5e-8
patience = 10
optimizer = tf.optimizers.Adam(learning_rate=lr) 

def extract_features(X, fs, datapoint_dur, nfft = 4096, hop_length=512):

    X = np.array([[[x[i:i+int(datapoint_dur*fs)]] for i in range(0,len(x)-5*int(datapoint_dur*fs),int(datapoint_dur*fs))] for x in X]).transpose(1,2,3,0)
    X = X/np.max(X[:,edge:,edge:],axis=tuple(range(1,X.ndim)),keepdims=True)
    return X

# obtaining the training data
edge = 0
marco_data = {m:{} for m in marco_dataset.array_names}
marco_data = {'Ambeo':{},'Eigenmike':{}}
for array in marco_data.keys():
    for ss in ['single_sources-90d','single_sources-75d','single_sources-60d','single_sources-45d','single_sources-30d','single_sources-15d']: 
        print('getting ', array, ' ', ss)
        X = marco_dataset.get_audio_numpy(ss, array)

        # extract features
        marco_data[array][ss] = extract_features(X, fs, datapoint_dur)

# CNN parameters
nchans = 64
K = 33
ch1 = nchans*1
F1 = tf.Variable(tf.initializers.HeNormal()(shape=(1,K,3,ch1)))
b1 = tf.Variable(tf.constant_initializer(0.0)(shape=(ch1)))
ch2 = ch1
F2 = tf.Variable(tf.initializers.HeNormal()(shape=(1,K,ch1,ch2)))
b2 = tf.Variable(tf.constant_initializer(0.0)(shape=(ch2)))
ch3 = nchans*2
F3 = tf.Variable(tf.initializers.HeNormal()(shape=(1,K,ch2,ch3)))
b3 = tf.Variable(tf.constant_initializer(0.0)(shape=(ch3)))
ch4 = ch3
F4 = tf.Variable(tf.initializers.HeNormal()(shape=(1,K,ch3,ch4)))
b4 = tf.Variable(tf.constant_initializer(0.0)(shape=(ch4)))
ch5 = nchans*4
F5 = tf.Variable(tf.initializers.HeNormal()(shape=(1,K,ch4,ch5)))
b5 = tf.Variable(tf.constant_initializer(0.0)(shape=(ch5)))
ch6 = ch5
F6 = tf.Variable(tf.initializers.HeNormal()(shape=(1,K,ch5,ch6)))
b6 = tf.Variable(tf.constant_initializer(0.0)(shape=(ch6)))
ch7 = nchans*8
F7 = tf.Variable(tf.initializers.HeNormal()(shape=(1,K,ch6,ch7)))
b7 = tf.Variable(tf.constant_initializer(0.0)(shape=(ch7)))
ch8 = ch7
F8 = tf.Variable(tf.initializers.HeNormal()(shape=(1,K,ch7,ch8)))
b8 = tf.Variable(tf.constant_initializer(0.0)(shape=(ch8)))
ch9 = nchans*16
F9 = tf.Variable(tf.initializers.HeNormal()(shape=(1,K,ch8,ch9)))
b9 = tf.Variable(tf.constant_initializer(0.0)(shape=(ch9)))
ch10 = ch9
F10 = tf.Variable(tf.initializers.HeNormal()(shape=(1,K,ch9,ch10)))
b10 = tf.Variable(tf.constant_initializer(0.0)(shape=(ch10)))
ch11 = ch8
F11 = tf.Variable(tf.initializers.HeNormal()(shape=(1,K,ch11,ch10)))
b11 = tf.Variable(tf.constant_initializer(0.0)(shape=(ch11)))
F11b = tf.Variable(tf.initializers.HeNormal()(shape=(1,1,ch11*2,ch11*2)))
b11b = tf.Variable(tf.constant_initializer(0.0)(shape=(ch11*2)))
ch12 = ch8
F12 = tf.Variable(tf.initializers.HeNormal()(shape=(1,K,ch11*2,ch12)))
b12 = tf.Variable(tf.constant_initializer(0.0)(shape=(ch12)))
ch13 = ch8
F13 = tf.Variable(tf.initializers.HeNormal()(shape=(1,K,ch12,ch13)))
b13 = tf.Variable(tf.constant_initializer(0.0)(shape=(ch13)))
ch14 = ch6
F14 = tf.Variable(tf.initializers.HeNormal()(shape=(1,K,ch14,ch13)))
b14 = tf.Variable(tf.constant_initializer(0.0)(shape=(ch14)))
F14b = tf.Variable(tf.initializers.HeNormal()(shape=(1,1,ch14*2,ch14*2)))
b14b = tf.Variable(tf.constant_initializer(0.0)(shape=(ch14*2)))
ch15 = ch6
F15 = tf.Variable(tf.initializers.HeNormal()(shape=(1,K,ch14*2,ch15)))
b15 = tf.Variable(tf.constant_initializer(0.0)(shape=(ch15)))
ch16 = ch6
F16 = tf.Variable(tf.initializers.HeNormal()(shape=(1,K,ch15,ch16)))
b16 = tf.Variable(tf.constant_initializer(0.0)(shape=(ch16)))
ch17 = ch4
F17 = tf.Variable(tf.initializers.HeNormal()(shape=(1,K,ch17,ch16)))
b17 = tf.Variable(tf.constant_initializer(0.0)(shape=(ch17)))
F17b = tf.Variable(tf.initializers.HeNormal()(shape=(1,1,ch17*2,ch17*2)))
b17b = tf.Variable(tf.constant_initializer(0.0)(shape=(ch17*2)))
ch18 = ch4
F18 = tf.Variable(tf.initializers.HeNormal()(shape=(1,K,ch17*2,ch18)))
b18 = tf.Variable(tf.constant_initializer(0.0)(shape=(ch18)))
ch19 = ch4
F19 = tf.Variable(tf.initializers.HeNormal()(shape=(1,K,ch18,ch19)))
b19 = tf.Variable(tf.constant_initializer(0.0)(shape=(ch19)))
ch20 = ch2
F20 = tf.Variable(tf.initializers.HeNormal()(shape=(1,K,ch20,ch19)))
b20 = tf.Variable(tf.constant_initializer(0.0)(shape=(ch20)))
F20b = tf.Variable(tf.initializers.HeNormal()(shape=(1,1,ch20*2,ch20*2)))
b20b = tf.Variable(tf.constant_initializer(0.0)(shape=(ch20*2)))
ch21 = ch2
F21 = tf.Variable(tf.initializers.HeNormal()(shape=(1,K,ch20*2,ch21)))
b21 = tf.Variable(tf.constant_initializer(0.0)(shape=(ch21)))
ch22 = ch2
F22 = tf.Variable(tf.initializers.HeNormal()(shape=(1,K,ch21,ch22)))
b22 = tf.Variable(tf.constant_initializer(0.0)(shape=(ch22)))
ch23 = 1
F23 = tf.Variable(tf.initializers.GlorotNormal()(shape=(1,K,ch22,ch23)))
b23 = tf.Variable(tf.constant_initializer(0.0)(shape=(ch23)))

# capsule coordinates network parameters
sH1 = 32
sW1 = tf.Variable(tf.initializers.HeNormal()(shape=(3,sH1)))
sb1 = tf.Variable(tf.constant_initializer(0.0)(shape=(sH1,)))
sH2 = 64
sW2 = tf.Variable(tf.initializers.HeNormal()(shape=(sH1,sH2)))
sb2 = tf.Variable(tf.constant_initializer(0.0)(shape=(sH2,)))
sH3 = K*K*(ch1)
sW3 = tf.Variable(tf.initializers.HeNormal()(shape=(sH2,sH3)))
sb3 = tf.Variable(tf.constant_initializer(0.0)(shape=(sH3,)))

# optimizer variables
mse = tf.losses.MeanSquaredError()
all_vars = [sW1,sb1,sW2,sb2,sW3,sb3,F1,F2,F3,F4,F5,F6,F7,F8,F9,F10,F11,F12,F13,F14,F15,F16,F17,F18,F19,F20,F21,F22,F23,F11b,F14b,F17b,F20b,b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11,b12,b13,b14,b15,b16,b17,b18,b19,b20,b21,b22,b23,b11b,b14b,b17b,b20b]

# The network architecture
@tf.function()
def forward_pass(X,Xc,Yc,all_vars,N,H,W,chans_in,ch1,droprate):

    # unpack variables
    sW1,sb1,sW2,sb2,sW3,sb3,F1,F2,F3,F4,F5,F6,F7,F8,F9,F10,F11,F12,F13,F14,F15,F16,F17,F18,F19,F20,F21,F22,F23,F11b,F14b,F17b,F20b,b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11,b12,b13,b14,b15,b16,b17,b18,b19,b20,b21,b22,b23,b11b,b14b,b17b,b20b = all_vars

    # process input capsule coordinates
    #sh1 = tf.nn.relu(tf.add(tf.matmul(Xc, sW1), sb1))
    #sh2 = tf.nn.relu(tf.add(tf.matmul(sh1, sW2), sb2))
    #sh3 = tf.add(tf.matmul(sh2, sW3), sb3)
    #F = tf.reshape(sh3, (K,K,chans_in,ch1))

    # process output chapsule coordinates
    #sf1 = tf.nn.relu(tf.add(tf.matmul(Yc, sW1), sb1))
    #sf2 = tf.nn.relu(tf.add(tf.matmul(sf1, sW2), sb2))
    #Fo = tf.add(tf.matmul(sf2, sW3), sb3)
    #Fo = tf.reshape(Fo, (K,K,ch1,1))
    
    # combine filters
    F1d = F1 #tf.repeat(F1,chans_in,axis=2)
    F23d = F23

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
best_MSE = float("inf")
patience_count = 0
final_epoch = 0
epoch = 0
best_vars = []
while True:
    if epoch == 0:
        print("====================================")
        print("====================================")
        print("====================================")
        print("Training set error (before training)")
    elif epoch==1:
        print("====================================")
        print(":::::::   Training Starts   ::::::::")
        print("====================================")
    elif epoch==final_epoch and epoch > patience:
        print("====================================")
        print("Training done!")
        print("Training set error w best parameters")
        print("====================================")
        all_vars = best_vars
    elif epoch==final_epoch+1 and epoch > patience:
        print("====================================")
        print("Evaluation after training")
        print("====================================")
        marco_data = {m:{} for m in marco_dataset.array_names}
        for array in marco_data.keys():
            for ss in ['single_sources0deg']:
                print('getting ', array, ' ', ss)
                X = marco_dataset.get_audio_numpy(ss,array)
        
                # extract features
                marco_data[array][ss] = extract_features(X, fs, datapoint_dur)
        

    epoch_error = 0
    for array in marco_data.keys():

        coords, capsule_names = marco_dataset.get_capsule_coords_numpy(array)

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
                    coords_cart = np.array([v for v in _polar2cart({j:i for i,j in zip(coords,capsule_names)}, units='radians').values()])
                    target_chan = np.array(coords_cart[ichan])
                    coords_cart[ichan] = np.array([1000,1000,1000])
                    nbrs = NearestNeighbors(n_neighbors=3).fit(coords_cart)
                    _, all_chans_in = nbrs.kneighbors([target_chan])
                    ncombs = len(all_chans_in)
                    if ichan in all_chans_in:
                        raise ValueError('something is really really wrong!')
                    
                    comb_error = 0
                    for chans_in in all_chans_in:

                        batch_size = 512
                        batch_idx = np.random.choice(data.shape[0],batch_size,replace=False)
                        X = data[batch_idx][...,np.array(chans_in)]
                        Xc = coords[np.array(chans_in)].astype(np.float32)
                        Y = data[batch_idx][:,edge:,edge:,[ichan]]
                        Yc = coords[[ichan]].astype(np.float32)
                        N, H, W, chans_in = X.shape

                        if epoch > 0 and patience_count < patience:
                            with tf.GradientTape() as g:
                                out= forward_pass(X,Xc,Yc,all_vars,N,H,W,chans_in,ch1,0.0)
                                error = mse(out,Y)
                                comb_error += error.numpy()
                            gradients = g.gradient(error,all_vars[6:])
                            optimizer.apply_gradients(zip(gradients,all_vars[6:]))
                        else:
                            out = forward_pass(X,Xc,Yc,all_vars,N,H,W,chans_in,ch1,0)
                            error = mse(out,Y)
                            comb_error += error.numpy()
                    total_error += comb_error/ncombs 
                if patience_count >= patience:
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
                    if epoch == final_epoch:
                        plt.savefig('_'.join(['plots/train',ss,array,str(in_size),'.png']))
                    if epoch == final_epoch+1:
                        plt.savefig('_'.join(['plots/eval',ss,array,str(in_size),'.png']))
                    plt.close()
                in_size_error += total_error/nchans 
                if epoch > 0 and patience_count <= patience:
                    print('epoch:', epoch, array, ss,'No. input chans',in_size, 'MSE:', total_error/nchans)
                else:
                    print(array,ss,'No. input chans',in_size, 'MSE:', total_error/nchans)
            source_error += in_size_error/len(range(nchans-1,nchans))

        epoch_error += source_error/len(marco_data[array].keys())
    print('-----------------------------')
    print('                             ')
    print('   (Best MSE so far:', best_MSE,')')
    print('Average error was', epoch_error/len(marco_data.keys()))
    print('                             ')
    print('-----------------------------')
    if epoch == 0:
        best_MSE = epoch_error/len(marco_data.keys())
    elif epoch == final_epoch+1 and patience_count >= patience:
        break
    elif patience_count < patience and epoch_error/len(marco_data.keys())>=best_MSE:
        patience_count += 1
        if patience_count >= patience:
            print(" ")
            print("PATIENCE ELAPSED")
            print(" ")
            final_epoch = epoch
            patience_count += 1
            continue
    elif epoch_error/len(marco_data.keys()) < best_MSE and epoch > 0 and final_epoch<epoch:
        best_MSE = epoch_error/len(marco_data.keys())
        print("****************************")
        print("NEW best epoch error found!")
        print('Best MSE so far:', best_MSE)
        if patience_count > 0:
            print("saving parameters")
            [np.save(''.join(['best_params/parameter_',str(i),'.npy']),a.numpy()) for i,a in enumerate(all_vars)]
            best_vars = [tf.identity(var) for var in all_vars]
        print("****************************")
        patience_count = 0
    epoch += 1

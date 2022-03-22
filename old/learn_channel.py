import librosa
import numpy as np
import tensorflow as tf
from itertools import chain, combinations
from micarraylib.datasets import marco
import matplotlib.pyplot as plt

marco_dataset = marco(download=False, data_home='/home/iran/datasets/marco')
marco_dataset.fs = 8000
fs = marco_dataset.fs

# CNN parameters
K = 32
ch1 = 16 
F1 = tf.Variable(tf.initializers.HeNormal()(shape=(K,K,1,ch1)))
b1 = tf.Variable(tf.constant_initializer(0.0)(shape=(ch1)))
ch2 = 32
F2 = tf.Variable(tf.initializers.HeNormal()(shape=(K,K,ch1,ch2)))
b2 = tf.Variable(tf.constant_initializer(0.0)(shape=(ch2)))
ch3 = 64
F3 = tf.Variable(tf.initializers.HeNormal()(shape=(K,K,ch2,ch3)))
b3 = tf.Variable(tf.constant_initializer(0.0)(shape=(ch3)))
ch4 = ch3
F4 = tf.Variable(tf.initializers.HeNormal()(shape=(K,K,ch4,ch3)))
b4 = tf.Variable(tf.constant_initializer(0.0)(shape=(ch4)))
ch5 = ch2
F5 = tf.Variable(tf.initializers.HeNormal()(shape=(K,K,ch5,ch4)))
b5 = tf.Variable(tf.constant_initializer(0.0)(shape=(ch5)))
ch6 = ch1
F6 = tf.Variable(tf.initializers.HeNormal()(shape=(K,K,ch6,ch5)))
b6 = tf.Variable(tf.constant_initializer(0.0)(shape=(ch6)))
ch7 = 1
F7 = tf.Variable(tf.initializers.HeNormal()(shape=(K,K,ch6,ch7)))
b7 = tf.Variable(tf.constant_initializer(0.0)(shape=(ch7)))

# capsule coordinates network parameters
sH1 = 32
sW1 = tf.Variable(tf.initializers.HeNormal()(shape=(3,sH1)))
sb1 = tf.Variable(tf.initializers.HeNormal()(shape=(sH1,)))
sH2 = 64
sW2 = tf.Variable(tf.initializers.HeNormal()(shape=(sH1,sH2)))
sb2 = tf.Variable(tf.initializers.HeNormal()(shape=(sH2,)))
sH3 = K*K*(ch1)
sW3 = tf.Variable(tf.initializers.HeNormal()(shape=(sH2,sH3)))
sb3 = tf.Variable(tf.initializers.HeNormal()(shape=(sH3,)))
nnw = tf.Variable(tf.constant_initializer(0.0)(shape=(1,)))

# optimizer variables
lr = 1e-6
optimizer = tf.optimizers.Adam(learning_rate=lr) 
mse = tf.losses.MeanSquaredError()
all_vars = [sW1,sb1,sW2,sb2,sW3,sb3,F1,F2,F3,F4,F5,F6,F7,b1,b2,b3,b4,b5,b6,b7,nnw]

# obtaining the data
marco_data = {m:{} for m in marco_dataset.array_names}
del marco_data['Eigenmike']
for array in marco_data.keys():
    for ss in ['impulse_response-90d','impulse_response-45d','impulse_response+45d','impulse_response+90d']:
        print('getting ', array, ' ', ss)
        X = marco_dataset.get_audio_numpy(ss, array)

        # extract features
        nfft = 1024
        X = np.array([np.abs(librosa.stft(x, n_fft=nfft, hop_length=nfft//2)) for x in X]).transpose(1,2,0)
        X = 20*np.log10(X) 
        X = (X - np.mean(X,axis=tuple(range(0,X.ndim-1))))/np.std(X,axis=tuple(range(0,X.ndim-1)))
        marco_data[array][ss] = X 

@tf.function()
def forward_pass(X,Xc,Yc,all_vars,N,H,W,chans_in,ch1):

    # unpack variables
    sW1,sb1,sW2,sb2,sW3,sb3,F1,F2,F3,F4,F5,F6,F7,b1,b2,b3,b4,b5,b6,b7,nnw = all_vars

    # process input capsule coordinates
    sh1 = tf.nn.relu(tf.add(tf.matmul(Xc, sW1), sb1))
    sh2 = tf.nn.relu(tf.add(tf.matmul(sh1, sW2), sb2))
    sh3 = tf.add(tf.matmul(sh2, sW3), sb3)
    F = tf.reshape(sh3, (K,K,chans_in,ch1))

    # process output chapsule coordinates
    sf1 = tf.nn.relu(tf.add(tf.matmul(Yc, sW1), sb1))
    sf2 = tf.nn.relu(tf.add(tf.matmul(sf1, sW2), sb2))
    Fo = tf.add(tf.matmul(sf2, sW3), sb3)
    Fo = tf.reshape(Fo, (K,K,ch1,1))
    
    # combine filters
    F1p = tf.repeat(F1,chans_in,axis=2)*tf.nn.softmax(F,axis=2) + 0*nnw*F
    F7p = F7*tf.nn.softmax(Fo,axis=2) + 0*nnw*Fo

    # pass 
    out = tf.nn.conv2d(X,F1p,[1,1,1,1],padding='SAME')
    out1 = tf.nn.relu(tf.nn.bias_add(out,b1))
    out = tf.nn.max_pool2d(out1, ksize=4, strides=2, padding="SAME")
    out = tf.nn.conv2d(out, F2, [1,1,1,1], padding="SAME")
    out2 = tf.nn.relu(tf.nn.bias_add(out,b2))
    out = tf.nn.max_pool2d(out2, ksize=4, strides=2, padding="SAME")
    out = tf.nn.conv2d(out, F3, [1,1,1,1], padding="SAME")
    out3 = tf.nn.relu(tf.nn.bias_add(out,b3))
    out = tf.nn.max_pool2d(out3, ksize=4, strides=2, padding="SAME")
    out = tf.nn.conv2d_transpose(out, F4, out3.shape, [1,2,2,1], padding="SAME")
    out = tf.nn.relu(tf.nn.bias_add(out,b4))
    out = tf.nn.conv2d_transpose(out, F5, out2.shape, [1,2,2,1], padding="SAME")
    out = tf.nn.relu(tf.nn.bias_add(out,b5))
    out = tf.nn.conv2d_transpose(out, F6, out1.shape, [1,2,2,1], padding="SAME")
    out = tf.nn.relu(tf.nn.bias_add(out,b6))
    out = tf.nn.conv2d(out, F7p, [1,1,1,1], padding="SAME")
    out = tf.nn.relu(tf.nn.bias_add(out,b7))
    return out

def training_error():
    print("====================================")
    print("====================================")
    print("====================================")
    print("Training set error (before training)")
    # error before training
    epoch_error = 0
    for array in marco_data.keys():
    
        coords, _ = marco_dataset.get_capsule_coords_numpy(array)
    
        for ss in marco_data[array].keys():
    
            data = marco_data[array][ss]
            nchans = data.shape[-1]
    
            for in_size in reversed(range(nchans-2,nchans)):
    
                total_error = 0
                for ichan in range(nchans):
    
                    chan_range = list(range(nchans))
                    chan_range.remove(ichan)
    
                    all_chans_in = combinations(chan_range, in_size)
                    ncombs = sum(1 for i in all_chans_in)
                    all_chans_in = combinations(chan_range, in_size)
    
                    for chans_in in all_chans_in:
                        
                        X = data[...,np.array(chans_in)][np.newaxis]
                        Xc = coords[np.array(chans_in)].astype(np.float32)
                        Y = data[...,[ichan]][np.newaxis]
                        Yc = coords[[ichan]].astype(np.float32)
                        N, H, W, chans_in = X.shape
                      
                        out = forward_pass(X,Xc,Yc,all_vars,N,H,W,chans_in,ch1)
                        error = mse(out,Y)
                        total_error += error.numpy()
                    total_error /= ncombs 
                total_error /= nchans 
                epoch_error += total_error
                print(array, 'with ',ss,'and',in_size, 'in chans error was: ', total_error)
            epoch_error /= len(range(nchans-2,nchans))
        epoch_error /= len(marco_data[array].keys())
    print('Total error was', epoch_error)
    print("filter weight is", all_vars[-1].numpy())

training_error()

best_MSE = float("inf")
for epoch in range(1000):
    epoch_error = 0
    for array in marco_data.keys():

        coords, _ = marco_dataset.get_capsule_coords_numpy(array)

        for ss in marco_data[array].keys():

            data = marco_data[array][ss]
            nchans = data.shape[-1]

            for in_size in reversed(range(nchans-2,nchans)):

                total_error = 0
                for ichan in range(nchans):

                    chan_range = list(range(nchans))
                    chan_range.remove(ichan)

                    all_chans_in = combinations(chan_range, in_size)
                    ncombs = sum(1 for i in all_chans_in)
                    all_chans_in = combinations(chan_range, in_size)

                    for chans_in in all_chans_in:

                        X = data[...,np.array(chans_in)][np.newaxis]
                        Xc = coords[np.array(chans_in)].astype(np.float32)
                        Y = data[...,[ichan]][np.newaxis]
                        Yc = coords[[ichan]].astype(np.float32)
                        N, H, W, chans_in = X.shape

                        with tf.GradientTape() as g:
                            out= forward_pass(X,Xc,Yc,all_vars,N,H,W,chans_in,ch1)
                            error = mse(out,Y)
                            total_error += error.numpy()
                        gradients = g.gradient(error,all_vars)
                        optimizer.apply_gradients(zip(gradients,all_vars))
                    total_error /= ncombs 
                total_error /= nchans 
                epoch_error += total_error
                print('epoch: ', epoch, array, 'with ',in_size, 'input chans. MSE: ', total_error)
            epoch_error /= len(range(nchans-2,nchans))
        epoch_error /= len(marco_data[array].keys())
    print('Epoch ', epoch, 'error was', epoch_error)
    print("filter weight is", all_vars[-1].numpy())
    if epoch_error < best_MSE:
        best_MSE = epoch_error
        print("****************************")
        print("NEW best epoch error found!")
        print("saving parameters")
        [np.save(''.join(['parameter_',str(i),'.npy']),a.numpy()) for i,a in enumerate(all_vars)]
        print("****************************")

training_error()

# Evaluation error
marco_data = {m:{} for m in marco_dataset.array_names}
for array in marco_data.keys():
    for ss in ['impulse_response0deg']:
        print('getting ', array, ' ', ss)
        X = marco_dataset.get_audio_numpy(ss,array)

        # extract features
        nfft = 1024
        X = np.array([np.abs(librosa.stft(x, n_fft=nfft, hop_length=nfft//2)) for x in X]).transpose(1,2,0)
        X = 20*np.log10(X) 
        X = (X - np.mean(X,axis=tuple(range(0,X.ndim-1))))/np.std(X,axis=tuple(range(0,X.ndim-1)))
        marco_data[array][ss] = X 

print("====================")
print("====================")
print("====================")
print("Evaluation set error")
# error after training
for array in marco_data.keys():

    coords, _ = marco_dataset.get_capsule_coords_numpy(array)

    for ss in marco_data[array].keys():

        data = marco_data[array][ss]
        nchans = data.shape[-1]

        for in_size in reversed(range(nchans-2,nchans)):

            total_error = 0
            for ichan in range(nchans):

                chan_range = list(range(nchans))
                chan_range.remove(ichan)

                all_chans_in = combinations(chan_range, in_size)
                ncombs = sum(1 for i in all_chans_in)
                all_chans_in = combinations(chan_range, in_size)

                for chans_in in all_chans_in:

                    X = data[...,np.array(chans_in)][np.newaxis]
                    Xc = coords[np.array(chans_in)].astype(np.float32)
                    Y = data[...,[ichan]][np.newaxis]
                    Yc = coords[[ichan]].astype(np.float32)
                    N, H, W, chans_in = X.shape

                    out = forward_pass(X,Xc,Yc,all_vars,N,H,W,chans_in,ch1)
                    error = mse(out,Y)
                    total_error += error.numpy()

                fig, axs = plt.subplots(1, 2, figsize=(10, 5))
                axs[0].imshow(np.squeeze(Y),aspect='auto')
                axs[1].imshow(np.squeeze(out),aspect='auto')
                plt.savefig('_'.join(['eval',ss,array,str(in_size),'.png']))
                plt.close()
                total_error /= ncombs 
            total_error /= nchans 
            print('array: ', array, 'with ',in_size, 'input channels error was: ', total_error)

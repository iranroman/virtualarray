import librosa
import numpy as np
import tensorflow as tf
from itertools import chain, combinations
from micarraylib.datasets import marco
import matplotlib.pyplot as plt

marco_dataset = marco(download=False, data_home='/home/iran/datasets/marco')
fs = marco_dataset.fs

channels_img = 1
out_channels = 32
K = 8

# supplementary network parameters
sH1 = 32
sW1 = tf.Variable(tf.initializers.HeNormal()(shape=(1,3,sH1)))
sb1 = tf.Variable(tf.initializers.HeNormal()(shape=(1,sH1)))
sH2 = 64
sW2 = tf.Variable(tf.initializers.HeNormal()(shape=(1,sH1,sH2)))
sb2 = tf.Variable(tf.initializers.HeNormal()(shape=(1,sH2)))
sH3 = K*K*(out_channels)
sW3 = tf.Variable(tf.initializers.HeNormal()(shape=(1,sH2,sH3)))
sb3 = tf.Variable(tf.initializers.HeNormal()(shape=(1,sH3)))

b1 = tf.Variable(tf.constant_initializer(0.0)(shape=(out_channels)))
F2 = tf.Variable(tf.initializers.HeNormal()(shape=(K,K,out_channels,32)))
b2 = tf.Variable(tf.constant_initializer(0.0)(shape=(32)))
F3 = tf.Variable(tf.initializers.HeNormal()(shape=(K,K,32,64)))
b3 = tf.Variable(tf.constant_initializer(0.0)(shape=(64)))
F4 = tf.Variable(tf.initializers.HeNormal()(shape=(K,K,64,64)))
b4 = tf.Variable(tf.constant_initializer(0.0)(shape=(64)))
F5 = tf.Variable(tf.initializers.HeNormal()(shape=(K,K,32,64)))
b5 = tf.Variable(tf.constant_initializer(0.0)(shape=(32)))
F6 = tf.Variable(tf.initializers.HeNormal()(shape=(K,K,out_channels,32)))
b6 = tf.Variable(tf.constant_initializer(0.0)(shape=(out_channels)))
b7 = tf.Variable(tf.constant_initializer(0.0)(shape=(1)))
optimizer = tf.optimizers.Adam(learning_rate=0.001) 
mse = tf.losses.MeanSquaredError()

all_vars = [sW1,sb1,sW2,sb2,sW3,sb3,F2,F3,F4,F5,F6,b1,b2,b3,b4,b5,b6,b7]

marco_data = {m:{} for m in marco_dataset.array_names}
for array in marco_data.keys():
    if array == 'Eigenmike':
        continue
    for ss in ['impulse_response-90d','impulse_response-45d','impulse_response+45d','impulse_response+90d']:#marco_dataset.sound_sources:
        print('getting ', array, ' ', ss)
        X = marco_dataset.get_audio_numpy(array, ss)

        # extract features
        nfft = 1024
        X = np.array([X[:,i*fs:i*fs+fs] for i in range(X.shape[1]//fs)])
        X = np.array([[np.abs(librosa.stft(i, n_fft=nfft, hop_length=nfft//2)) for i in x] for x in X])
        marco_data[array][ss] = X

@tf.function()
def forward_pass(X,Xc,Yc,all_vars,N,MB,H,W,out_channels,channels_img):

    sW1,sb1,sW2,sb2,sW3,sb3,F2,F3,F4,F5,F6,b1,b2,b3,b4,b5,b6,b7 = all_vars

    sh1 = tf.nn.relu(tf.add(tf.matmul(Xc, sW1), sb1))
    sh2 = tf.nn.relu(tf.add(tf.matmul(sh1, sW2), sb2))
    sh3 = tf.add(tf.matmul(sh2, sW3), sb3)
    sf1 = tf.nn.relu(tf.add(tf.matmul(Yc, sW1), sb1))
    sf2 = tf.nn.relu(tf.add(tf.matmul(sf1, sW2), sb2))
    fo = tf.add(tf.matmul(sf2, sW3), sb3)
    F = tf.reshape(sh3, (MB,K,K,1,out_channels))
    fo = tf.reshape(fo, (K,K,out_channels,1))
    
    MB, fh, fw, channels, out_channels = F.shape
    
    # F has shape (MB, fh, fw, channels, out_channels)
    # REM: with the notation in the question, we need: channels_img==channels
    
    Fd = tf.transpose(F, [1, 2, 0, 3, 4])
    Fd = tf.reshape(Fd, [fh, fw, channels*MB, out_channels])
    
    inp_r = tf.transpose(X, [1, 2, 0, 3]) # shape (H, W, MB, channels_img)
    inp_r = tf.reshape(inp_r, [1, H, W, MB*channels_img])
    
    out = tf.nn.depthwise_conv2d(
              inp_r,
              filter=Fd,
              strides=[1, 1, 1, 1],
              padding="SAME") # here no requirement about padding being 'VALID', use whatever you want. 
    
    out = tf.reshape(out, [H, W, MB, channels, out_channels])
    out = tf.transpose(out, [2, 0, 1, 3, 4])
    out = tf.reduce_sum(out, axis=3)
    out = tf.reduce_sum(out,axis=0,keepdims=True)
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
    out = tf.nn.conv2d(out, fo, [1,1,1,1], padding="SAME")
    out = tf.nn.relu(tf.nn.bias_add(out,b7))
    return out

print("====================================")
print("====================================")
print("====================================")
print("Training set error (before training)")
# error before training
for array in marco_data.keys():

    coords = np.array([v for v in marco_dataset.capsule_coords[array].values()])

    for ss in marco_data[array].keys():

        data = marco_data[array][ss]
        nchans = data.shape[1]

        for in_size in reversed(range(nchans-2,nchans)):

            total_error = 0
            for ichan in range(nchans):
                #print('array ', array, 'in_size: ', in_size, 'out of ',len(coords),'ichan_out: ', ichan)

                chan_range = list(range(nchans))
                chan_range.remove(ichan)

                all_chans_in = combinations(chan_range, in_size)
                ncombs = sum(1 for i in all_chans_in)
                all_chans_in = combinations(chan_range, in_size)

                for chans_in in all_chans_in:

                    X = data[:,np.array(chans_in)]
                    Xc = coords[np.array(chans_in)][np.newaxis,:].astype(np.float32)
                    Y = data[:,ichan]
                    Yc = coords[[ichan]]
                    N, MB, H, W = X.shape

                    for i in range(N):
                  
                        # supplementary network operations
                        out = forward_pass(X[i][...,np.newaxis],Xc,Yc[np.newaxis,...].astype(np.float32),all_vars,N,MB,H,W,out_channels,channels_img)
                        error = mse(out,Y[i][np.newaxis,...,np.newaxis])
                        total_error += error.numpy()
                    total_error /= N 
                total_error /= ncombs 
            total_error /= nchans 
            print(array, 'with ',ss,'source and',in_size, 'in chans error was: ', total_error)

for epoch in range(100):

    for array in marco_data.keys():

        coords = np.array([v for v in marco_dataset.capsule_coords[array].values()])

        for ss in marco_data[array].keys():

            data = marco_data[array][ss]
            nchans = data.shape[1]

            for in_size in reversed(range(nchans-2,nchans)):

                total_error = 0
                for ichan in range(nchans):
                    #print('array ', array, 'in_size: ', in_size, 'out of ',len(coords),'ichan_out: ', ichan)

                    chan_range = list(range(nchans))
                    chan_range.remove(ichan)

                    all_chans_in = combinations(chan_range, in_size)
                    ncombs = sum(1 for i in all_chans_in)
                    all_chans_in = combinations(chan_range, in_size)

                    for chans_in in all_chans_in:

                        X = data[:,np.array(chans_in)]
                        Xc = coords[np.array(chans_in)][np.newaxis,:].astype(np.float32)
                        Y = data[:,ichan]
                        Yc = coords[[ichan]]
                        N, MB, H, W = X.shape

                        for i in range(N):
                      
                            with tf.GradientTape() as g:
                                # supplementary network operations
                                out = forward_pass(X[i][...,np.newaxis],Xc,Yc[np.newaxis,...].astype(np.float32),all_vars,N,MB,H,W,out_channels,channels_img)
                                error = mse(out,Y[i][np.newaxis,...,np.newaxis])
                                total_error += error.numpy()
                            gradients = g.gradient(error,all_vars)
                            optimizer.apply_gradients(zip(gradients,all_vars))
                        total_error /= N 
                    total_error /= ncombs 
                total_error /= nchans 
                print('epoch: ', epoch, array, 'with ',in_size, 'input chans. MSE: ', total_error)

print("====================")
print("====================")
print("====================")
print("Training set error")
# error after training
for array in marco_data.keys():

    coords = np.array([v for v in marco_dataset.capsule_coords[array].values()])

    for ss in marco_data[array].keys():

        data = marco_data[array][ss]
        nchans = data.shape[1]

        for in_size in reversed(range(nchans-2,nchans)):

            total_error = 0
            for ichan in range(nchans):
                #print('array ', array, 'in_size: ', in_size, 'out of ',len(coords),'ichan_out: ', ichan)

                chan_range = list(range(nchans))
                chan_range.remove(ichan)

                all_chans_in = combinations(chan_range, in_size)
                ncombs = sum(1 for i in all_chans_in)
                all_chans_in = combinations(chan_range, in_size)

                for chans_in in all_chans_in:

                    X = data[:,np.array(chans_in)]
                    Xc = coords[np.array(chans_in)][np.newaxis,:].astype(np.float32)
                    Y = data[:,ichan]
                    Yc = coords[[ichan]]
                    N, MB, H, W = X.shape

                    for i in range(N):
                  
                        # supplementary network operations
                        out = forward_pass(X[i][...,np.newaxis],Xc,Yc[np.newaxis,...].astype(np.float32),all_vars,N,MB,H,W,out_channels,channels_img)
                        error = mse(out,Y[i][np.newaxis,...,np.newaxis])
                        total_error += error.numpy()
                        if i==0 and ichan == nchans//2:
                            fig, axs = plt.subplots(1, 2, figsize=(10, 5))
                            axs[0].imshow(Y[i],aspect='auto')
                            axs[1].imshow(np.squeeze(out),aspect='auto')
                            plt.savefig('_'.join(['train',ss,array,str(in_size),'.png']))
                            plt.close()
                            total_error /= N 
                total_error /= ncombs 


            total_error /= nchans 
            print('array: ', array, 'with ',in_size, 'input channels error was: ', total_error)

marco_data = {m:{} for m in marco_dataset.array_names}
for array in marco_data.keys():
    for ss in ['impulse_response0deg']:
        print('getting ', array, ' ', ss)
        X = marco_dataset.get_audio_numpy(array, ss)

        # extract features
        nfft = 1024
        X = np.array([X[:,i*fs:i*fs+fs] for i in range(X.shape[1]//fs)])
        X = np.array([[np.abs(librosa.stft(i, n_fft=nfft, hop_length=nfft//2)) for i in x] for x in X])
        marco_data[array][ss] = X

print("====================")
print("====================")
print("====================")
print("Evaluation set error")
# error after training
for array in marco_data.keys():

    coords = np.array([v for v in marco_dataset.capsule_coords[array].values()])

    for ss in marco_data[array].keys():

        data = marco_data[array][ss]
        nchans = data.shape[1]

        for in_size in reversed(range(nchans-2,nchans)):

            total_error = 0
            for ichan in range(nchans):
                #print('array ', array, 'in_size: ', in_size, 'out of ',len(coords),'ichan_out: ', ichan)

                chan_range = list(range(nchans))
                chan_range.remove(ichan)

                all_chans_in = combinations(chan_range, in_size)
                ncombs = sum(1 for i in all_chans_in)
                all_chans_in = combinations(chan_range, in_size)

                for chans_in in all_chans_in:

                    X = data[:,np.array(chans_in)]
                    Xc = coords[np.array(chans_in)][np.newaxis,:].astype(np.float32)
                    Y = data[:,ichan]
                    Yc = coords[[ichan]]
                    N, MB, H, W = X.shape

                    for i in range(N):
                  
                        # supplementary network operations
                        out = forward_pass(X[i][...,np.newaxis],Xc,Yc[np.newaxis,...].astype(np.float32),all_vars,N,MB,H,W,out_channels,channels_img)
                        error = mse(out,Y[i][np.newaxis,...,np.newaxis])
                        if i==0 and ichan == nchans//2:
                            fig, axs = plt.subplots(1, 2, figsize=(10, 5))
                            axs[0].imshow(Y[i],aspect='auto')
                            axs[1].imshow(np.squeeze(out),aspect='auto')
                            plt.savefig('_'.join(['eval',ss,array,str(in_size),'.png']))
                            plt.close()
                        total_error += error.numpy()
                    total_error /= N 
                total_error /= ncombs 
            total_error /= nchans 
            print('array: ', array, 'with ',in_size, 'input channels error was: ', total_error)

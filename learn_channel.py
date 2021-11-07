import librosa
import numpy as np
import tensorflow as tf
from itertools import chain, combinations
from micarraylib.datasets import marco

marco_dataset = marco(download=False, data_home='/home/iran/datasets/marco')
fs = marco_dataset.fs

channels_img = 1
out_channels = 4
K = 30

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

F2 = tf.Variable(tf.initializers.HeNormal()(shape=(K,K,out_channels,1)))
F3 = tf.Variable(tf.initializers.HeNormal()(shape=(K,K,1,out_channels)))
optimizer = tf.optimizers.Adam(learning_rate=0.0001) 
mse = tf.losses.MeanSquaredError()

all_vars = [sW1,sb1,sW2,sb2,sW3,sb3,F2,F3]

marco_data = {m:{} for m in marco_dataset.array_names}
for array in ['Eigenmike']:#marco_data.keys():
    for ss in ['impulse_response+90d']:#marco_dataset.sound_sources:
        print('getting ', array, ' ', ss)
        X = marco_dataset.get_audio_numpy(array, ss)

        # extract features
        nfft = 1024
        X = np.array([X[:,i*fs:i*fs+fs] for i in range(X.shape[1]//fs)])
        marco_data[array][ss] = np.array([[np.abs(librosa.stft(i, n_fft=nfft, hop_length=nfft//2)) for i in x] for x in X])
        

for epoch in range(2000):

    for array in marco_data.keys():

        coords = np.array([v for v in marco_dataset.capsule_coords[array].values()])

        for ss in marco_data[array].keys():

            data = marco_data[array][ss]
            nchans = data.shape[1]

            for in_size in range(nchans-2,nchans-1):

                total_error = 0
                for ichan in range(nchans):
                    print('in_size: ', in_size, 'ichan_out: ', ichan)

                    chan_range = list(range(nchans))
                    chan_range.remove(ichan)

                    all_chans_in = combinations(chan_range, in_size)
                    ncombs = sum(1 for i in all_chans_in)
                    all_chans_in = combinations(chan_range, in_size)

                    for chans_in in all_chans_in:
                        X = data[:,np.array(chans_in)]
                        Xc = coords[np.array(chans_in)][np.newaxis,:].astype(np.float32)
                        Y = data[:,ichan]
                        Yc = coords[ichan]
                        N, MB, H, W = X.shape

                        for i in range(N):
                        
                            with tf.GradientTape() as g:
                                # supplementary network operations
                                sh1 = tf.nn.relu(tf.add(tf.matmul(Xc, sW1), sb1))
                                sh2 = tf.nn.relu(tf.add(tf.matmul(sh1, sW2), sb2))
                                sh3 = tf.add(tf.matmul(sh2, sW3), sb3)
                                sf1 = tf.nn.relu(tf.add(tf.matmul(Yc[np.newaxis,...].astype(np.float32), sW1), sb1))
                                sf2 = tf.nn.relu(tf.add(tf.matmul(sf1, sW2), sb2))
                                fo = tf.add(tf.matmul(sf2, sW3), sb3)
                                F = tf.reshape(sh3, (MB,K,K,1,out_channels))
                                fo = tf.reshape(fo, (K,K,1,out_channels))
                                
                                MB, fh, fw, channels, out_channels = F.shape
                                
                                # F has shape (MB, fh, fw, channels, out_channels)
                                # REM: with the notation in the question, we need: channels_img==channels
                                
                                Fd = tf.transpose(F, [1, 2, 0, 3, 4])
                                Fd = tf.reshape(Fd, [fh, fw, channels*MB, out_channels])
                                
                                inp_r = tf.transpose(X[i][...,np.newaxis], [1, 2, 0, 3]) # shape (H, W, MB, channels_img)
                                inp_r = tf.reshape(inp_r, [1, H, W, MB*channels_img])
                                
                                out = tf.nn.depthwise_conv2d(
                                          inp_r,
                                          filter=Fd,
                                          strides=[1, 1, 1, 1],
                                          padding="SAME") # here no requirement about padding being 'VALID', use whatever you want. 
                                
                                out = tf.reshape(out, [H, W, MB, channels, out_channels])
                                out = tf.transpose(out, [2, 0, 1, 3, 4])
                                out = tf.reduce_sum(out, axis=3)
                                out1 = tf.nn.relu(tf.reduce_sum(out,axis=0,keepdims=True))
                                out = tf.nn.max_pool2d(out1, ksize=4, strides=2, padding="SAME")
                                out = tf.nn.relu(tf.nn.conv2d(out, F2, [1,1,1,1], padding="SAME"))
                                out = tf.nn.relu(tf.nn.conv2d(out, fo, [1,1,1,1], padding="SAME"))
                                out = tf.nn.conv2d_transpose(out, F3, out1.shape, [1,2,2,1], padding="SAME")
                                error = mse(out,Y[i][np.newaxis,...,np.newaxis])
                                total_error += error.numpy()
                            gradients = g.gradient(error,all_vars)
                            optimizer.apply_gradients(zip(gradients,all_vars))
                        total_error /= N 
                    total_error /= ncombs 
                total_error /= nchans 
                print('epoch: ', epoch, 'array: ', array, 'with ',in_size, 'input channels error was: ', total_error)

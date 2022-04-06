import numpy as np
import tensorflow.keras

class DataGenerator(tensorflow.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size=120, dim=(32,32,32), n_channels=1, tetras_dict=None):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.data_arrays = [np.load('/scratch/irr2020/unet/data_npy/eigenscape/{}.npy'.format(f)) for f in list_IDs]
        self.n_channels = n_channels
        self.clips_per_npy = 5000
        self.on_epoch_end()
        capsules_in = []
        capsules_out = []
        for K, V in tetras_dict.items():
            for v in V:
                capsules_in.extend(np.array([int(j)-1 for j in v]))
                capsules_out.extend([int(K)-1])
        self.capsules_in = np.array([capsules_in])
        self.capsules_out = np.array([capsules_out])

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(len(self.list_IDs)*self.clips_per_npy)

    def __getitem__(self, index):
        'Load one file of data'
        # one file at a time
        npy_index = index//self.clips_per_npy
        clip_index = index%self.clips_per_npy
        data = self.data_arrays[npy_index][clip_index]

        # Generate data
        X, y = self.__data_generation(data)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs)*self.clips_per_npy)

    def __data_generation(self, data):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        
        X = np.take_along_axis(data, self.capsules_in, axis=-1).reshape(1920,self.batch_size,3)
        y = np.take_along_axis(data, self.capsules_out, axis=-1).T

        return X[:,np.newaxis].transpose(2, 1, 0, 3), y[:,np.newaxis,:,np.newaxis]

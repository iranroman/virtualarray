import numpy as np
import pickle

with open('tetras_dict.pkl', 'rb') as f:
    tetras_dict = pickle.load(f)


vl_filenames = [
    'Beach-01-Raw',
    'BusyStreet-04-Raw',
    'PedestrianZone-06-Raw',
    'QuietStreet-04-Raw',
    'ShoppingCentre-06-Raw',
    'TrainStation-04-Raw',
    'Woodland-07-Raw',
]

X = np.concatenate([np.load('../data_npy/eigenscape/{}.npy'.format(f)) for f in vl_filenames], axis=0)
print(X.shape)
    
capsules_in = []
capsules_out = []
for K, V in tetras_dict.items():
    for v in V:
        capsules_in.extend(np.array([int(j)-1 for j in v]))
        capsules_out.extend([int(K)-1])
capsules_in = np.array([capsules_in])
capsules_out = np.array([capsules_out])

run_sum = 0
for i, data in enumerate(X):
    x = np.take_along_axis(data, capsules_in, axis=-1).reshape(1920,120,3)
    y = np.take_along_axis(data, capsules_out, axis=-1)

    run_sum += np.mean(np.square(np.mean(x,axis=-1)-y))

    if i % 1000 == 0:
        print(run_sum/(i+1))

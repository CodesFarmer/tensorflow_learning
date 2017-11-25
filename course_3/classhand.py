import h5py
import numpy as np

h5file =  h5py.File('/home/slam/TestRoom/python/test24_hand.h5', 'r+')
dataset = h5file['/label']
# data_read32 = np.zeros(shape=[None, 5], dtype=np.float32)
# dataset.id.read(h5py.h5s.ALL, h5py.h5s.ALL, data_read32, mtype=h5py.h5t.NATIVE_FLOAT)
label = dataset.value
print(label.shape)
h5file.close()
import h5py
import numpy as np

h5file =  h5py.File('/home/slam/TestRoom/python/test24_hand.h5', 'r+')
dataset_label = h5file['/label']
dataset_data = h5file['/data']
# data_read32 = np.zeros(shape=[None, 5], dtype=np.float32)
# dataset.id.read(h5py.h5s.ALL, h5py.h5s.ALL, data_read32, mtype=h5py.h5t.NATIVE_FLOAT)
label = dataset_label.value
data = dataset_data.value
index = np.random.permutation(label.shape[0])
label = label[index, :]
data = data[index, :]
batch_size = 128
num_batch = int(np.floor(data.shape[0]/batch_size))
for i in range(1, num_batch):
    index_start = (num_batch - 1)*batch_size
    index_end = num_batch*batch_size - 1
h5file.close()
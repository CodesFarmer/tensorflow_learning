import h5py
import numpy as np
import tensorflow as tf

h5file =  h5py.File('/home/slam/nfs70/rnn_sh/workspace/tools/data/test12_hand.h5', 'r+')
dataset_label = h5file['/label']
dataset_data = h5file['/data']
# data_read32 = np.zeros(shape=[None, 5], dtype=np.float32)
# dataset.id.read(h5py.h5s.ALL, h5py.h5s.ALL, data_read32, mtype=h5py.h5t.NATIVE_FLOAT)
label = dataset_label.value
data = dataset_data.value
index = np.random.permutation(label.shape[0])
# label = label[index, :]
# data = data[index, :]
batch_size = 128
num_batch = int(np.floor(data.shape[0]/batch_size))
print(data.shape)

#transfer label to tensor
ltensor = tf.convert_to_tensor(label[:, 0], tf.float32)
lmatrix = tf.one_hot(tf.cast(ltensor, tf.int32), 3)
two = tf.constant(0, dtype=tf.float32)
where = tf.equal(ltensor, two)
# index = tf.cast(tf.lin_space(0.0, int(ltensor.get_shape()[0]), int(ltensor.get_shape()[0])), tf.int32)
# index = tf.multiply(where, index)
# index = tf.tenso
sess = tf.Session()
position = tf.Session().run(tf.boolean_mask(lmatrix, where))
# position = sess.run(index)
print(position)



for i in range(1, num_batch):
    index_start = (num_batch - 1)*batch_size
    index_end = num_batch*batch_size - 1
h5file.close()

# h5file = h5py.File('/tmp/cifar10_data/CIFAR-10.hdf5')
# print(h5file.attrs['std'])
# print(h5file.attrs['mean'])
# train_data = h5file['/normalized_full/training/default']
# print(train_data.shape)
# train_label = h5file['/normalized_full/training/targets']
# print(train_label.shape)
#
# for i in range(10):
#     print(i)
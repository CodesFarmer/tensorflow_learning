from six import string_types, iteritems
import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

#define the model_fn, where our model are designed
def layers(op):
    def layer_decorated(self, *args, **kwargs):
        '''
        All layers are abstracted to same operation
        input -> process -> output = input -> process -> output
        So we abstract the process here and realize it in each layer
        '''
        #set the layer name
        name = kwargs.setdefault('name', self.get_unique_name(op.__name__))
        print('Name in layer_decorated: %r'%name)
        #set the layer's input
        if len(self.terminals) == 0:
            raise RuntimeError('Empty input for layer %s'%name)
        elif len(self.terminals) == 1:
            layer_inputs = self.terminals[0]
        else:
            layer_inputs = list(self.terminals)
        print("The args is %r"%kwargs)
        layer_outputs = op(self, layer_inputs, *args, **kwargs)
        self.layers[name] = layer_outputs
        self.feed(layer_outputs)
        self.logits = []
        self.logits = layer_outputs
        return self
    return layer_decorated
class NeuralNetwork(object):
    def __init__(self, inputs, trainable=True):
        self.inputs = inputs
        #terminals store the intermediate result,
        # which is the output of last layer and the input of next layer
        self.terminals = []
        self.layers = dict(inputs)
        self.trainable = trainable
        self.logits = []
        self.setup()
    def setup(self):
        raise NotImplementedError('Must be implemented by sub class')
    def get_unique_name(self, prefix):
        ident = sum(t.startswith(prefix) for t,_ in self.layers.items()) + 1
        return '%s_%d'%(prefix, ident)
    def feed(self, *args):
        '''
        This function look up the dictionary of layers by the layer name
        '''
        assert len(args)!=0
        self.terminals = []
        for arg in args:
            if isinstance(arg, string_types):
                try:
                    arg = self.layers[arg]
                except KeyError:
                    raise KeyError('Unknow layer name %s'%arg)
            self.terminals.append(arg)
        return self
    def make_variable(self, name, shape):
        # tf.truncated_normal_initializer
        initialization = tf.truncated_normal(shape=shape, stddev=0.1, dtype=tf.float32)
        return tf.get_variable(name=name, initializer=initialization)
        # return tf.get_variable(name=name, shape=shape, initializer=tf.truncated_normal_initializer)
    @layers
    def conv(self, input_nn, k_h, k_w, s_h, s_w, channels, name, padding='VALID'):
        #get the input depth
        input_dim = int(input_nn.get_shape()[-1])
        #define the variables for convolution
        convolue = lambda inp, kernel_nn: tf.nn.conv2d(inp, kernel_nn, [1, s_h, s_w, 1], padding=padding)
        with tf.variable_scope(name) as scope:
            #In tensorflow, the convolutional kernel is [h, w, in_channels, out_channels]
            kernel_nn = self.make_variable('weights', [k_h, k_w, input_dim, channels])
            output = convolue(input_nn, kernel_nn)
            #add the biases
            biases = self.make_variable('biases', [channels])
            output = tf.add(output, biases)
            return output

    @layers
    def activate(self, input_nn, name, atype_nn='PReLU'):
        with tf.variable_scope(name) as scope:
            if atype_nn.lower() == 'relu':
                output = tf.nn.relu(input_nn, name=scope.name)
                return output
            elif atype_nn.lower() == 'sigmoid':
                output = tf.nn.sigmoid(input_nn, name=scope.name)
                return output
            elif atype_nn.lower() == 'prelu':
                i = int(input.get_shape()[-1])
                alpha = self.make_variable('alpha', shape=(i,))
                output = tf.nn.relu(input_nn) + tf.multiply(alpha, -tf.nn.relu(-input_nn))
                return output
            else:
                raise RuntimeError('Unknow activations: %s'%atype_nn)

    @layers
    def fc(self, input_nn, output_num, name):
        #get the input shape
        input_shape = input_nn.get_shape()
        input_num = 1
        for num in input_shape[1:].as_list():
            input_num = input_num*int(num)
        with tf.variable_scope(name) as scope:
            W = self.make_variable('weights', [input_num, output_num])
            biases = self.make_variable('biases', [output_num])
            input_flatten = tf.reshape(input_nn, [-1, input_num])
            output = tf.add(tf.matmul(input_flatten, W), biases)
            return output

    @layers
    def pool(self, input_nn, k_h, k_w, s_h, s_w, name, ptype_nn='MAX', padding='SAME'):
        with tf.variable_scope(name) as scope:
            if ptype_nn.lower() == 'max':
                output = tf.nn.max_pool(input_nn, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)
                return output
            elif ptype_nn.lower() == 'avg':
                output = tf.nn.avg_pool(input_nn, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)
                return output
            else:
                raise KeyError('Unknow pooling kernel %s'%ptype_nn)

    @layers
    def dropout(self, input_nn, keep_prob, name):
        with tf.variable_scope(name):
            output = tf.nn.dropout(input_nn, keep_prob=keep_prob)
            return output

class mnist_nn(NeuralNetwork):
    def setup(self):
        (self.feed('data')
         .conv(5, 5, 1, 1, 32, name='conv1', padding='SAME')
         .activate(name='relu1', atype_nn='relu')
         .pool(2, 2, 2, 2, name='pool1')
         .conv(5, 5, 1, 1, 64, name='conv2', padding='SAME')
         .activate(name='relu2', atype_nn='relu')
         .pool(2, 2, 2, 2, name='pool2')
         # .dropout(keep_prob=0.5, name='drop1')
         .fc(1024, name='fc1')
         .activate(name='relu3', atype_nn='relu')
         .dropout(keep_prob=0.5, name='drop2')
         .fc(10, name='fc2')
         )

# def create_cnn(sess):
#     with tf.variable_scope('mnist'):
#         input_x = tf.placeholder(dtype=tf.float32, shape=[None, 784], name='input')
#         data = tf.reshape(input_x, [-1, 28, 28, 1])
#         #{'data':data} is a kwargs, the keyword is 'data'
#         # the corresponding variable is stored in data
#         mnist_nn({'data': data})
#     mnist_fun = lambda imgs : sess.run(('mnist/fc2/fc2:0'), feed_dict={'mnist/input:0': imgs})
#     return mnist_fun

#defien the loss
print('Let\'s flying~')
x = tf.placeholder(dtype=tf.float32, shape=[None, 784])
y = tf.placeholder(dtype=tf.float32, shape=[None, 10])
sess = tf.Session()
# mnist_cnn = create_cnn(sess)
# logits = ('mnist/fc2/fc2:0')
data = tf.reshape(x, [-1, 28, 28, 1])
logits = mnist_nn({'data': data}).logits
# logits = mnist_nn({'data': data}).layers['fc2']
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))
train_op = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
# train_op = tf.train.AdamOptimizer(1e-4).minimize(loss)
init = tf.global_variables_initializer()
sess.run(init)
for i in range(1, 10000):
    batch = mnist.train.next_batch(50)
    # sess.run(train_op, {x: batch[0], y: batch[1]})
    sess.run(train_op, feed_dict={x: batch[0], y: batch[1]})
    if i%100 == 0:
        print("%d iterations-loss : %r"%(i, sess.run(loss, feed_dict={x: batch[0], y: batch[1]})))
        corrections = tf.equal(tf.argmax(y, 1), tf.argmax(logits, 1))
        accuracy = tf.reduce_mean(tf.cast(corrections, tf.float32))
        print("TEST: The final accuracy on test is %r" % sess.run(accuracy, feed_dict={
            x: mnist.test.images[0:5000, :], y: mnist.test.labels[0:5000, :]}))

'''
There are some experience:
1, Do not set the learning rate of Adam too high, otherwise it will become unconvergent
2, The proper initialization will help to optimize the neural network,
    do not just set the initializer = tf.truncated_normal_initializer,
     instead, you should set initializer=tf.truncated_normal(shape=shape, stddev=0.1, dtype=tf.float32)
3, The input data is important for you to realize your network, please make sure it is valid before feed into network
'''
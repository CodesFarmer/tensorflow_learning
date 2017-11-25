import tensorflow as tf
from six import string_types

'''
In this shell, we define a class of neural network
Which can support the basic layers in CNN
It looks like a basic class, users can extend it for your own network
'''
_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5

def layers(op):
    def layers_decorated(self, *args, **kwargs):
        #First of all, we should get the name of this layer
        #We set the default name of this layer
        name = kwargs.setdefault('name', self.get_unique_name(op.__name__))
        #Set the input for this layer, which are stored at intermediate
        if len(self.intermediate) == 0:
            raise RuntimeError('The input of layer %s is empty!'%name)
        elif len(self.intermediate) == 1:
            input_data = self.intermediate[0]
        else:
            input_data = list(self.intermediate)

        #pass the data through the layer
        output_data = op(self, input_data, *args, **kwargs)
        #Bind the output of this layer to layers
        self.layers[name] = output_data
        #feed the output of this layer to next layer, we put the data into intermediate actually
        self.feed(output_data)
        return self
    return layers_decorated

class NeuralNetwork(object):
    def __init__(self, input_nn):
        #set the input of the neural network
        self.input_ = input_nn
        #set the outputs of intermediate layer, which are used for connect layers
        self.intermediate = []
        #set the name of layer
        self.layers = dict(input_nn)
        #set the state of the neural network(TRAIN OR TEST)
        self.training = True
        #the setup() function, which define the layers
        self.setup()
    #If the subclass does not define the setup function, we throw a Error
    def setup(self):
        raise RuntimeError('You must realize setup function youself')
    #Define a function that can get the name of the layer
    def get_unique_name(self, prefix):
        #First we count the number of layers have the same type to prefix
        layer_id = sum(t.startswith(prefix) for t, _ in self.layers.items())
        return '%s_%d'%(prefix, layer_id+1)
    #define the function for making variables
    def make_variables(self, name, shape, initializer='GAUSSIAN'):
        if initializer.lower() == 'truncated':
            initialization = tf.truncated_normal(shape=shape, mean=0.0, stddev=0.1)
        elif initializer.lower() == 'zeros':
            initialization = tf.zeros(shape=shape)
        elif initializer.lower() == 'gamma':
            initialization = tf.random_gamma(shape=shape, alpha=1.5, beta=2.0)
        else:
            # raise RuntimeWarning('Initialization method %s does not support'%initializer)
            initialization = tf.random_normal(shape=shape, mean=0.0, stddev=0.1)
        return tf.get_variable(name=name, initializer=initialization)
    def feed(self, *args):
        '''
        :param args:
        :return: set the intermediate
        '''
        assert len(args) != 0
        #clear the intermediate
        self.intermediate = []
        for arg in args:
            #get the layer name
            if isinstance(arg, string_types):
                try:
                    arg = self.layers[arg]
                except KeyError:
                    raise KeyError('Unknow layer name %s'%arg)
            self.intermediate.append(arg)
        return self
    def feed_multi(self, operation = 'concat', *args):
        #set the input
        '''
        We support multi input with specified operation
        :param operation: ADD, CONCATENATION
        :param args: the corresponding input
        :return:
        '''
        assert len(args) != 0
        #clear the intermediate at first
        self.intermediate = []
        #get the shape of input data
        data_container = []
        for arg in args:
            #get the layer name
            if isinstance(arg, string_types):
                try:
                    layer_out_data = self.layers[arg]
                except:
                    raise KeyError('Unknow layer name %s'%arg)
            if operation.lower() == 'concat':
                if data_container == []:
                    data_container = layer_out_data
                else:
                    data_container = tf.concat([data_container, layer_out_data], 3)
            elif operation.lower() == 'add':
                if data_container == []:
                    data_container = tf.zeros(shape=layer_out_data.get_shape(), dtype=tf.float32)
                elif data_container.get_shape() != layer_out_data.get_shape():
                    raise RuntimeError('The shape of layer %s does not equal to the others'%arg)
                #If those two layers have same dimension, we add it
                data_container = tf.add(data_container, layer_out_data)
        self.intermediate = data_container
        return self

    @layers
    def conv(self, input_nn, k_h, k_w, s_h, s_w, out_channels, name, padding='VALID', initializer='GAUSSIAN'):
        #We get the depth of last feature map
        in_channels = int(input_nn.get_shape()[-1])
        #We define the convolutional function
        convolue = lambda x, kernel: tf.nn.conv2d(x, kernel, [1, s_h, s_w, 1], padding=padding)
        with tf.variable_scope(name) as scope:
            #define the weights in convolutional layer
            weight = self.make_variables(name='weight', shape=[k_h, k_w, in_channels, out_channels], initializer=initializer)
            bias = self.make_variables(name='bias', shape=[out_channels])
            output = convolue(input_nn, weight)
            output = tf.add(output, bias)
            return output
    @layers
    def bottleneck_block(self, input_nn, out_channels_1, out_channels_2, out_channels_3, name, is_training=True,
                         initializer='GAUSSIAN', activation='ReLU'):
        #define the convolutional layer
        convolue = lambda x, kernel: tf.nn.conv2d(x, kernel, [1, 1, 1, 1], padding='SAME')
        #This is a block of resnet, which is consist of three convolutional layers
        in_channels = int(input_nn.get_shape()[-1])
        assert in_channels == out_channels_3
        short_cut = input_nn
        input_nn = self.batch_norm(input_nn, name='%s_bn'%name, is_training=is_training)
        with tf.variable_scope(name):
            #First layer, which is 1x1 convolution
            weight_1 = self.make_variables(name='weight_1', shape=[1, 1, in_channels, out_channels_1], initializer=initializer)
            bias_1 = self.make_variables(name='bias_1', shape=[out_channels_1])
            output_1 = tf.add(convolue(input_nn, weight_1), bias_1)
            output_1 = self.batch_norm(output_1, name='bn_1', is_training=is_training)
            output_1 = self.activate_intra(output_1, name='actv_1', activation=activation)
            #Second layer, which is 3x3 convolution
            weight_2 = self.make_variables(name='weight_2', shape=[3, 3, out_channels_1, out_channels_2], initializer=initializer)
            bias_2 = self.make_variables(name='bias_2', shape=[out_channels_2])
            output_2 = tf.add(convolue(output_1, weight_2), bias_2)
            output_2 = self.batch_norm(output_2, name='bn_2', is_training=is_training)
            output_2 = self.activate_intra(output_2, name='actv_2', activation=activation)
            #Third layer, which is 1x1 convolution
            weight_3 = self.make_variables(name='weight_3', shape=[1, 1, out_channels_2, out_channels_3], initializer=initializer)
            bias_3 = self.make_variables(name='bias_3', shape=[out_channels_3])
            output_3 = tf.add(convolue(output_2, weight_3), bias_3)
            output_3 = self.batch_norm(output_3, name='bn_3', is_training=is_training)
            return self.activate_intra(tf.add(short_cut, output_3), name='actv', activation=activation)

    def batch_norm(self, input_nn, name, is_training=True):
        with tf.variable_scope(name):
            output = tf.layers.batch_normalization(inputs=input_nn, axis=3,
                                                   momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON,
                                                   center=True, scale=True, training=is_training, name=name, reuse=False, fused=True)
            return output
    #Activating function
    def activate_intra(self, input_nn, name, activation='PReLU'):
        with tf.variable_scope(name) as scope:
            if activation.lower() == 'relu':
                output = tf.nn.relu(input_nn, name=scope.name)
                return output
            elif activation.lower() == 'sigmoid':
                output = tf.nn.sigmoid(input_nn, name=scope.name)
                return output
            elif activation.lower() == 'prelu':
                i = int(input.get_shape()[-1])
                alpha = self.make_variable('alpha', shape=(i,))
                output = tf.nn.relu(input_nn) + tf.multiply(alpha, -tf.nn.relu(-input_nn))
                return output
            else:
                raise RuntimeError('Unknow activations: %s'%activation)
    #Activating function
    @layers
    def activate(self, input_nn, name, activation='PReLU'):
        with tf.variable_scope(name) as scope:
            if activation.lower() == 'relu':
                output = tf.nn.relu(input_nn, name=scope.name)
                return output
            elif activation.lower() == 'sigmoid':
                output = tf.nn.sigmoid(input_nn, name=scope.name)
                return output
            elif activation.lower() == 'prelu':
                i = int(input.get_shape()[-1])
                alpha = self.make_variable('alpha', shape=(i,))
                output = tf.nn.relu(input_nn) + tf.multiply(alpha, -tf.nn.relu(-input_nn))
                return output
            else:
                raise RuntimeError('Unknow activations: %s'%activation)

    #define the FC layer
    @layers
    def fc(self, input_nn, out_channels, name, initializer='GAUSSIAN'):
        #Get the input dimension of this layer
        in_shape = input_nn.get_shape()
        in_dimension = 1
        for num_dim in in_shape[1:].as_list():
            in_dimension = in_dimension*int(num_dim)
        #add a fully connected layer
        with tf.variable_scope(name):
            weight = self.make_variables('weight', shape=[in_dimension, out_channels], initializer=initializer)
            bias = self.make_variables('bias', shape=[out_channels])
            #before multiple, we flat the matrix into vector
            featmap_flat = tf.reshape(input_nn, [-1, in_dimension])
            output = tf.add(tf.matmul(featmap_flat, weight), bias)
            return output
    #define the pooling layer
    @layers
    def pool(self, input_nn, k_h, k_w, s_h, s_w, name, ptype_nn='MAX', padding='SAME'):
        with tf.variable_scope(name):
            if ptype_nn.lower() == 'max':
                output = tf.nn.max_pool(input_nn, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)
                return output
            elif ptype_nn.lower() == 'avg':
                output = tf.nn.avg_pool(input_nn, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)
                return output
            else:
                raise KeyError('Unknow pooling kernel %s'%ptype_nn)
    #define the dropout layer
    @layers
    def dropout(self, input_nn, keep_prob, name):
        with tf.variable_scope(name):
            output = tf.nn.dropout(input_nn, keep_prob=keep_prob)
            return output
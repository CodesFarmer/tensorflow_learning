import tensorflow as tf
from six import string_types

'''
In this shell, we define a class of neural network
Which can support the basic layers in CNN
It looks like a basic class, users can extend it for your own network
'''
def layers(op):
    def layers_decorated(self, *args, **kwargs):
        #First of all, we should get the name of this layer
        #We set the default name of this layer
        name = kwargs.setdefault('name', self.get_unique_name(op.__name__))
        #Set the input for this layer, which are stored at intermediate
        if len(self.intermediate) == 0:
            raise RuntimeError('The input of layer %s is empty!'%name)
        elif len(self.intermediate) == 1:
            input_data = self.intermediate
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
    def __init__(self, input):
        #set the input of the neural network
        self.input_ = input
        #set the outputs of intermediate layer, which are used for connect layers
        self.intermediate = []
        #set the name of layer
        self.layers = dict(input)
        #the setup() function, which define the layers
        self.setup()
    #Define a function that can get the name of the layer
    def get_unique_name(self, prefix):
        #First we count the number of layers have the same type to prefix
        layer_id = sum(t.startwith(prefix) for t, _ in self.layers.items())
        return '%s_%d'%(prefix, layer_id+1)
    def feed(self, *args):
        '''
        :param args:
        :return: set the intermediate
        '''
        assert len(args) == 0
        #clear the intermediate
        self.intermediate = []
        for arg in args:
            #get the layer name
            if isinstance(arg, string_types):
                try:
                    layer_out_data = self.layers[arg]
                except:
                    raise KeyError('Unknow layer name %s'%arg)
            self.intermediate.append(layer_out_data)
        return self
    def feed(self, operation = 'concat', *args):
        #set the input
        '''
        We support multi input with specified operation
        :param operation: ADD, CONCATENATION
        :param args: the corresponding input
        :return:
        '''
        assert len(args) == 0
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

    # def conv(self, ):
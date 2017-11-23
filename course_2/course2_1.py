'''
In this program, we test the neural network in tensorflow
Again, we must redeclare that the important elements for tensorflow
1, Placeholder: data input interface(shape is important)
2, Trainable variables: weights in FC and Convolutional layers(initialization methods is important)
3, Loss: the objective function for optimizing the neural network should be designed appropriately
4, Optimizer: the optimization method should be specified
'''

import tensorflow as tf
import numpy as np

#import the minst data sets
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
#create a session, we do not know what the difference between Session and InteractiveSession
sess = tf.InteractiveSession()
#declare the inputs
x = tf.placeholder(dtype=tf.float32, shape=[None, 784])
y = tf.placeholder(dtype=tf.float32, shape=[None, 10])
#declare the weights and bias
layer1_num = 64
W1 = tf.get_variable(name="weights1", shape=[784, layer1_num], dtype=tf.float32, initializer=tf.random_normal_initializer)
b1 = tf.get_variable(name="bias1", shape=[layer1_num], dtype=tf.float32, initializer=tf.zeros_initializer)
W2 = tf.get_variable(name="weights2", shape=[layer1_num, 10], dtype=tf.float32, initializer=tf.random_normal_initializer)
b2 = tf.get_variable(name="bias2", shape=[10], dtype=tf.float32, initializer=tf.zeros_initializer)
z1 = tf.add(tf.matmul(x, W1), b1)
x2 = tf.nn.sigmoid(z1)
z2 = tf.add(tf.matmul(x2, W2), b2)
logits = z2
#define the loss
a1 = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)
loss = tf.reduce_mean(a1)
#define the optimizer
optimizer = tf.train.AdamOptimizer(0.01)
#optimizer = tf.train.GradientDescentOptimizer(0.05)
train_op = optimizer.minimize(loss)
#initialize the variables
init = tf.global_variables_initializer()
sess.run(init)
for i in range(1, 10000):
    batch = mnist.train.next_batch(100)
    sess.run(train_op, {x: batch[0], y: batch[1]})
    print("%d iterations-loss : %r"%(i, sess.run(loss, {x: batch[0], y: batch[1]})))

corrections = tf.equal(tf.argmax(y, 1), tf.argmax(logits, 1))
accuracy = tf.reduce_mean(tf.cast(corrections, tf.float32))
print("The final accuracy on test is %r"%sess.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels}))
'''
We use estimator in here for neural network for mnist
'''
import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def model_fn(features, labels, mode):
    #define the neural network
    W1 = tf.get_variable('Weights1', shape=[784, 64], dtype=tf.float32, initializer=tf.random_normal_initializer)
    b1 = tf.get_variable('bias1', shape=[64], dtype=tf.float32, initializer=tf.zeros_initializer)
    W2 = tf.get_variable('Weights2', shape=[64, 10], dtype=tf.float32, initializer=tf.random_normal_initializer)
    b2 = tf.get_variable('bias2', shape=[10], dtype=tf.float32, initializer=tf.zeros_initializer)
    z1 = tf.add(tf.matmul(features['x'], W1), b1)
    x2 = tf.sigmoid(z1)
    z2 = tf.add(tf.matmul(x2, W2), b2)
    logits = z2
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
    global_step = tf.train.get_global_step()
    optimizer = tf.train.AdamOptimizer(0.01)
    train_op = tf.group(optimizer.minimize(loss), tf.assign_add(global_step, 1))
    return tf.estimator.EstimatorSpec(
        mode = mode,
        predictions = logits,
        loss = loss,
        train_op=train_op
    )

estimator = tf.estimator.Estimator(model_fn=model_fn)
input_fn = tf.estimator.inputs.numpy_input_fn(
    {'x': mnist.train.images},
    mnist.train.labels,
    batch_size=100,
    num_epochs=None,
    shuffle=True
)
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    {'x': mnist.train.images},
    mnist.train.labels,
    batch_size=100,
    num_epochs=1000,
    shuffle=False
)
eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    {'x': mnist.test.images},
    mnist.test.labels,
    batch_size=100,
    num_epochs=1000,
    shuffle=False
)
#Training the neural network
estimator.train(input_fn=input_fn, steps=1000)
#Evaluations
train_metrics = estimator.evaluate(input_fn=train_input_fn)
eval_metrics = estimator.evaluate(input_fn=eval_input_fn)
print("Training informations: %r"%train_metrics)
print("Evaluating information: %r"%eval_metrics)
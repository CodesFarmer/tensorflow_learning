import tensorflow as tf
import neuralnetwork
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

class mnist_nn(neuralnetwork.NeuralNetwork):
    def setup(self):
        (self.feed('data')
         .conv(5, 5, 1, 1, 32, name='conv1', padding='SAME')
         .activate(name='relu1', activation='relu')
         .pool(2, 2, 2, 2, name='pool1')
         .bottleneck_block(8, name='res1a')
         .bottleneck_block(8, name='res2a')
         .bottleneck_block(8, name='res3a')
         # .activate(name='relu2', activation='relu')
         .pool(2, 2, 2, 2, name='pool2')
         .fc(1024, name='fc1')
         .activate(name='sigmoid1', activation='sigmoid')
         .fc(10, name='fc2')
        )

x = tf.placeholder(dtype=tf.float32, shape=[None, 784])
y = tf.placeholder(dtype=tf.float32, shape=[None, 10])
is_training = tf.placeholder(dtype=tf.bool)
data = tf.reshape(x, [-1, 28, 28, 1])
logits = mnist_nn({'data':data}).layers['fc2']
mnist_nn.training = is_training
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))
# optimizer = tf.train.GradientDescentOptimizer(0.1)
optimizer = tf.train.AdamOptimizer(0.0001)
train_op = optimizer.minimize(loss)
sess = tf.Session()
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
init = tf.global_variables_initializer()
sess.run(init)

for i in range(1, 10001):
    train_batch = mnist.train.next_batch(128)
    sess.run(train_op, {x:train_batch[0], y: train_batch[1], is_training: True})
    if i%500 == 0:
        print('The loss is %f'%sess.run(loss, {x:train_batch[0], y: train_batch[1], is_training: True}))
    if i%500 == 0:
        # print('Accuracy : %f'%sess.run(accuracy, {x:mnist.test.images[1:5000, :], y: mnist.test.labels[1:5000, :]}))
        print('Accuracy : %f'%sess.run(accuracy, {x:mnist.test.images, y: mnist.test.labels, is_training: False}))

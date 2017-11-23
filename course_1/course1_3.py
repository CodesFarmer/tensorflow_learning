import numpy as np
import tensorflow as tf

#define model_fn, which evaluate predictions, loss, training steps
def model_fn(features, labels, mode):
    #define a linear model and predict values
    #Note here we use get_variable instead of Variable
    #Which can get the existing variables in the scope "weight" and scope "bias"
    W = tf.get_variable("weight", [1], dtype=tf.float64)
    b = tf.get_variable("bias", [1], dtype=tf.float64)
    y = tf.add(tf.multiply(W, features['x']), b)
    loss = tf.reduce_mean(tf.square(labels - y))
    # optimizer = tf.train.AdamOptimizer(0.1)
    optimizer = tf.train.GradientDescentOptimizer(0.1)
    # train_op = optimizer.minimize(loss)
    global_step = tf.train.get_global_step()
    train_op = tf.group(optimizer.minimize(loss), tf.assign_add(global_step, 1))
    #Here we note that an estimator have some important elements
    #The output of neural network : predictions
    #The loss of neural network : loss
    #The optimization method : train_op
    #All of above is what we should realize ourselves
    return tf.estimator.EstimatorSpec(
        mode = mode,
        predictions = y,
        loss = loss,
        train_op = train_op
    )


#This line is equal to tf.estimator.LinearRegressor()
estimator = tf.estimator.Estimator(model_fn=model_fn)
#declare the data sets
x_train = np.array([1., 2., 3., 4.])
y_train = np.array([2., 5., 8., 11.])
x_eval = np.array([-1., 2.5, 6., 10.])
y_eval = np.array([-4., 6.5, 17., 29.])
#set the input interface
input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train}, y_train, batch_size=4, num_epochs=None, shuffle=True
)
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train}, y_train, batch_size=4, num_epochs=1000, shuffle=False
)
eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_eval}, y_eval, batch_size=4, num_epochs=1000, shuffle=False
)
#set the estimator
estimator.train(input_fn, steps=1000)
#set the evaluation methods
train_metrics = estimator.evaluate(input_fn=train_input_fn)
eval_metrics = estimator.evaluate(input_fn=eval_input_fn)
print("The training metrics : %r"%train_metrics)
print("The evaluation metrics : %r"%eval_metrics)
import tensorflow as tf
import numpy as np

#declare list of features
feature_columns = [tf.feature_column.numeric_column("x", shape=[1])]

#set a linear regression model
estimator = tf.estimator.LinearRegressor(feature_columns=feature_columns)

#declare the data sets
x_train = np.array([1., 2., 3., 4.])
y_train = np.array([2., 5., 8., 11.])
x_eval = np.array([-1., 2.5, 6., 10.])
y_eval = np.array([-4., 6.5, 17., 29.])

#define the input interface
input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train}, y_train, batch_size=4, num_epochs=None, shuffle=True
)

train_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train}, y_train, batch_size=4, num_epochs=1000, shuffle=False
)

eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_eval}, y_eval, batch_size=4, num_epochs=1000, shuffle=False
)

#set the training details
estimator.train(input_fn=input_fn, steps=10000)

#evaluation
train_metrics = estimator.evaluate(input_fn=train_input_fn)
eval_metrics = estimator.evaluate(input_fn=eval_input_fn)
#print the evaluation results
print("train metrics: %r"%train_metrics)
print("Evaluation metrics: %r"%eval_metrics)

'''
This example just illustrate the interface of estimator
which can be used to convenient for inputting data sets
'''
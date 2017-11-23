import tensorflow as tf

#declare two constant
node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0)
print(node1, node2)
#create a session object
sess = tf.Session()
print(sess.run([node1, node2]))
#add two node together
node3 = tf.add(node1, node2)
print("Node3 in tensorflow computional graph: ", node3)
print("The node3 through the graph: ", sess.run(node3))

#declare a palcehoder, which is a virtual variable will be fulled later
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = tf.add(a, b)
adder_and_multi = tf.multiply(adder_node, a)
adder_and_multi = tf.multiply(adder_and_multi, b)
print("The ontology in tensorflow computional graph: ", adder_and_multi)
#Now we can put some data through the graph
print("The output of graph while feed it some data: ", sess.run(adder_and_multi, {a:3, b:4}))
#Multi input for the graph
print("The output with multi input: ", sess.run(adder_and_multi, {a:[2, 5], b:[4, 6]}))

#In the following program, we just fit to a linear model
#declare the trainable variable W and b
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
#declare the input data placeholder
x = tf.placeholder(dtype=tf.float32)
y = tf.placeholder(dtype=tf.float32)
#define the linear model
linear_model = tf.add(tf.multiply(W, x), b)
#To fit the model, we must define a objective function, here we use squared error
square_error = tf.squared_difference(linear_model, y)
loss = tf.reduce_mean(square_error)
#Define an optimization method, and bind our loss to it
optimizer = tf.train.AdamOptimizer(0.1)
trainer = optimizer.minimize(loss)
#Initialize the trainable parameters
init = tf.global_variables_initializer()
sess.run(init)
#print the error before train
print("Loss is : ", sess.run(loss, {x:[1,2,3,4], y:[2,5,8,11]}))
for i in range(1,1000):
    sess.run(trainer, {x:[1,2,3,4], y:[2,5,8,11]})
    if(i%10 == 0):
        print("The loss at %d iterations: %f"%(i, sess.run(loss, {x:[1,2,3,4], y:[2,5,8,11]})))
print("The weight is %f, and the bias is %f"%(sess.run(W), sess.run(b)))

'''
Util now, we can have some conclusion:
1, It is important to understand the computational graph in tensorflow
   You can image it as an drink water system, we have a reservoir, which is our original data sets
   In the first sluice A, we filter out the most of fagots with coarse filter
   In the second sluice B, we precipitate some sediment...
   Then we set up the trivia during the purification, for example, during the processing,
    we should told the system which size of filter is should set up, 
    what kind of disinfectant we will use, how to control the temperature during distillation.
   Finally, we will have a detector to check if the water is cleaning, which must have a evaluation method.
   The water outputted finally may be unqualified, so we have to adjust the middle procession according to the output.
   
   All of those operations is similar to the convolution, max pooling, activations et.al. in CNN
   However, we should design the procession before we get through it.
   Concretely, we should design the structure of neural network(convolution, pooling, activation), 
    and how to adjust it(optimizer and loss), last, we would better set up the evaluation methods(validation or test)
    
   We put all of those operation in an online automatic system, like the session here,
     and the processing flow is computational graph, 
     the parameters during purification are similar to the trainable weights and bias in neural network.

   So it is important to understand the structure of tensorflow, there are several important elements in tensorlfow
   a, Session
   b, Variable
   c, Loss
   d, Optimizer

2, There some inevitable things we should do before we train a model
   a, Design each layers in neural network
   b, Declare the variables we will train
   c, Define the loss function
   d, Define the optimizer for training
   e, Initialize the variables
   f, Train the model with data sets
'''
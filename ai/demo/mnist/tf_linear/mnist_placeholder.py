import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import tensorflow as tf
#from tensorflow.examples.tutorials.mnist import input_data
#import input_data
from utils.datasets import input_data
import time

learning_rate = 0.01
batch_size = 100
n_epochs = 40

# 1. read in data
# using TF learn's built-in function to load MNIST data to the folder data/mnist
mnist = input_data.read_data_sets('../dataset', one_hot=True)
#X_batch, Y_batch = mnist.train.next_batch(batch_size)

# 2. create placeholders for feature and labels
# each image in the MNIST data is of shape 28*28 = 784
# therefore, each image is represented with a 1x784 tensor
# there are 10 classes for each image, corresponding to digits 0-9
# each label is one hot vector

#X = tf.placeholder(tf.float32, [batch_size, 784], name='image')
#Y = tf.placeholder(tf.int32, [batch_size, 10], name='label')

X = tf.placeholder(tf.float32, [None, 784], name='image')
Y = tf.placeholder(tf.int32, [None, 10], name='label')

# 3. create weights and bias
w = tf.get_variable(name='weights', shape=(784, 10), 
    initializer=tf.random_normal_initializer())
b = tf.get_variable(name='bias', shape=(1, 10),
    initializer=tf.zeros_initializer())

# 4. build model
logits = tf.matmul(X, w) + b

# 5. define loss function
# use cross entropy of softmax of logits as the loss function
entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y, 
    name='loss')
loss = tf.reduce_mean(entropy)
# loss = tf.reduce_mean(-tf.reduce_sum(tf.nn.softmax(logits) * tf.log(Y), reduction_indices=[1]))

# 6. define training op
# using gradient descent with learning rate of 0.01 to minimize loss
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# 7. calculate accuracy with test set
preds = tf.nn.softmax(logits)
correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

writer = tf.summary.FileWriter('./graphs/mnist_placeholder', tf.get_default_graph())
start_time = time.time()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    n_batches = int(mnist.train.num_examples/batch_size)
    
    # train the model n_epochs times
    for i in range(n_epochs):
        total_loss = 0
        for j in range(n_batches):
            X_batch, Y_batch = mnist.train.next_batch(batch_size)
            _, loss_batch = sess.run([optimizer, loss], {X:X_batch, Y:Y_batch})
            total_loss += loss_batch
        print('Average loss epoch {0}: {1}'.format(i, total_loss/n_batches))
    print('Total time: {0} seconds'.format(time.time() - start_time))

    # test the model
    n_batches = int(mnist.test.num_examples/batch_size)
    total_correct_preds = 0
    
    for i in range(n_batches):
        X_batch, Y_batch = mnist.test.next_batch(batch_size)
        accuracy_batch = sess.run(accuracy, {X:X_batch, Y:Y_batch})
        total_correct_preds += accuracy_batch

    print('Accuracy {0}'.format(total_correct_preds/mnist.test.num_examples))

writer.close()


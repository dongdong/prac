import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

from utils.datasets import input_data

DATA_PATH="../dataset/"

INPUT_NODE = 784
OUTPUT_NODE = 10

IMAGE_SIZE = 28
NUM_CHANNELS = 1
CONV1_DEEP = 32
CONV1_SIZE = 5
CONV2_DEEP = 64
CONV2_SIZE = 5
POOL_SIZE = 2
POOL_STRIDE = 2
FC_SIZE = 500

LEARNING_RATE = 0.01
TRAINING_STEPS = 10000
BATCH_SIZE = 100

def lenet5(inputs):
    inputs = tf.reshape(inputs, [-1, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS])
    net = slim.conv2d(inputs, CONV1_DEEP, [CONV1_SIZE, CONV1_SIZE],
            padding='SAME', scope='layer1-conv')
    net = slim.max_pool2d(net, POOL_SIZE, stride=POOL_STRIDE, 
            scope='layer2-max-pool')
    net = slim.conv2d(net, CONV2_DEEP, [CONV2_SIZE, CONV2_SIZE],
            padding='SAME', scope='layer3-conv')
    net = slim.max_pool2d(net, POOL_SIZE, stride=POOL_STRIDE,
            scope='layer4-max-pool')
    net = slim.flatten(net, scope='flatten')
    net = slim.fully_connected(net, FC_SIZE, scope='layer5')
    net = slim.fully_connected(net, OUTPUT_NODE, scope='output')
    return net

def train(mnist):
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')

    y = lenet5(x)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=y, labels=tf.argmax(y_, 1))
    loss = tf.reduce_mean(cross_entropy)
    train_op = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_value = sess.run([train_op, loss],
                feed_dict={x:xs, y_:ys})
            if i % 1000 == 0:
                print("After %d training step(s), loss on training "
                      "batch is %g." % (i, loss_value))

def main(argv=None):
    mnist = input_data.read_data_sets(DATA_PATH, one_hot=True)
    train(mnist)
            
if __name__ == '__main__':
    main()


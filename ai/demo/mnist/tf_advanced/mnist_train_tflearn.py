import tflearn
from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression

import tflearn.datasets.mnist as mnist

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
N_EPOCH = 20

train_X, train_Y, test_X, test_Y = mnist.load_data(
        data_dir=DATA_PATH, one_hot=True)

train_X = train_X.reshape([-1, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS])
test_X = test_X.reshape([-1, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS])

net = input_data(shape=[None, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS], name='input')
net = conv_2d(net, CONV1_DEEP, CONV1_SIZE, activation='relu')
net = max_pool_2d(net, POOL_SIZE)
net = conv_2d(net, CONV2_DEEP, CONV2_SIZE, activation='relu')
net = max_pool_2d(net, POOL_SIZE)
net = fully_connected(net, FC_SIZE, activation='relu')
net = fully_connected(net, OUTPUT_NODE, activation='softmax')

net = regression(net, optimizer='sgd', learning_rate=LEARNING_RATE,
        loss = 'categorical_crossentropy')

model = tflearn.DNN(net, tensorboard_verbose=0)
model.fit(train_X, train_Y, n_epoch=N_EPOCH,
        validation_set=([test_X, test_Y]),
        show_metric=True)


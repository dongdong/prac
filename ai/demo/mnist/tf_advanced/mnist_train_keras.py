import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras import backend as K
from utils.datasets import input_data
#import ssl
#ssl._create_default_https_context = ssl._create_unverified_context

DATA_PATH = '../dataset'

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

BATCH_SIZE = 128
N_EPOCH = 20

num_classes = OUTPUT_NODE
img_rows, img_cols = IMAGE_SIZE, IMAGE_SIZE

'''
(trainX, trainY), (testX, testY) = mnist.load_data(path=DATA_PATH)

if K.image_data_format == 'channels_first':
    trainX = trainX.reshape(trainX.shape[0], NUM_CHANNELS, img_rows, img_cols)
    testX = testX.reshpae(testX.shape[0], NUM_CHANNELS, img_rows, img_cols)
    input_shape = (NUM_CHANNELS, img_rows, img_cols)
else:
    trainX = TrainX.reshpe(trainX.shape[0], img_rows, img_cols, NUM_CHANNELS)
    testX = textX.reshape(testX.shape[0], img_rows, img_cols, NUM_CHANNELS)
    input_shape = (img_rows, img_cols, NUM_CHANNELS)

trainX = trainX.astype('float32')
testX = testX.astype('float32')
trainX /= 255.0
testX /= 255.0

trainY = keras.utils.to_categorical(trainY, num_classes)
testY = keras.utils.to_categorical(testY, num_classes)
'''
mnist = input_data.read_data_sets(DATA_PATH, one_hot=True)
 
trainX, trainY = mnist.train.images,mnist.train.labels
testX, testY = mnist.test.images, mnist.test.labels
trainX = trainX.reshape(-1, 28, 28,1).astype('float32')
testX = testX.reshape(-1,28, 28,1).astype('float32')
input_shape = (img_rows, img_cols, NUM_CHANNELS)

model = Sequential()
model.add(Conv2D(CONV1_DEEP, kernel_size=(CONV1_SIZE,CONV1_SIZE), 
        activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(POOL_SIZE, POOL_SIZE)))
model.add(Conv2D(CONV2_DEEP, (CONV2_SIZE, CONV2_SIZE),
        activation='relu'))
model.add(MaxPooling2D(pool_size=(POOL_SIZE, POOL_SIZE)))
model.add(Flatten())
model.add(Dense(FC_SIZE, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.SGD(),
        metrics=['accuracy'])


model.fit(trainX, trainY, batch_size=BATCH_SIZE, epochs=N_EPOCH,
        validation_data=(testX, testY))

score = model.evaluate(testX, testY)
print('Test loss:', score[0])
print('Test accuracy:', score[1])





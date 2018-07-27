import time

import tensorflow as tf
import tensorflow.contrib.eager as tfe

import utils

# In order to use eager execution
# called at the very begining of a TensorFlow program
tfe.enable_eager_execution()

DATA_FILE = 'data/birth_life_2010.txt'

# Read the data into a dataset
data, n_samples = utils.read_birth_life_data(DATA_FILE)
dataset = tf.data.Dataset.from_tensor_slices((data[:,0], data[:,1]))

# Create variables
w = tfe.Variable(0.0)
b = tfe.Variable(0.0)

# Define the linear predictor
def prediction(x):
    return x * w + b

# Define loss functions of the form: L(y, y_predicted)
def squared_loss(y, y_predicted):
    return (y - y_predicted) ** 2

def huber_loss(y, y_predicted, m=1.0):
    t = y - y_predicted
    return t ** 2 if tf.abs(t) <= m else m * (2 * tf.abs(t) - m)

def train(loss_fn):
    print('Training; loss function: ' + loss_fn.__name__)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    
    # Define the function through which to differentiate
    def loss_for_example(x, y):
        return loss_fn(y, prediction(x))
    
    grad_fn = tfe.implicit_value_and_gradients(loss_for_example)
    
    start = time.time()
    for epoch in range(100):
        total_loss = 0.0
        for x_i, y_i in tfe.Iterator(dataset):
            loss, gradients = grad_fn(x_i, y_i)
            optimizer.apply_gradients(gradients)
            total_loss += loss
        if epoch % 10 == 0:
            print('Epoch {0}:{1}'.format(epoch, total_loss / n_samples))

    print('Took %f seconds' % (time.time() - start))

train(huber_loss)
print(w, b)


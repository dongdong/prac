import numpy as np
import tensorflow as tf
from utils.datasets import input_data

DATA_PATH="../dataset/"

INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500

TRAINING_STEPS = 10000
BATCH_SIZE = 128
MODEL_DIR = "model/"

tf.logging.set_verbosity(tf.logging.INFO)

mnist = input_data.read_data_sets(DATA_PATH, one_hot=False)

feature_columns = [tf.feature_column.numeric_column('image', shape=[784])]

estimator = tf.estimator.DNNClassifier(
        feature_columns=feature_columns,
        hidden_units=[LAYER1_NODE],
        n_classes=OUTPUT_NODE,
        optimizer=tf.train.AdamOptimizer(),
        model_dir=MODEL_DIR)

train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'image': mnist.train.images},
        y=mnist.train.labels.astype(np.int32),
        num_epochs=None,
        batch_size=BATCH_SIZE,
        shuffle=True)
    
estimator.train(input_fn=train_input_fn, steps=TRAINING_STEPS)

test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'image': mnist.test.images},
        y=mnist.test.labels.astype(np.int32),
        num_epochs=1,
        batch_size=BATCH_SIZE,
        shuffle=False)

accuracy_score = estimator.evaluate(input_fn=test_input_fn)['accuracy']
print('\nTest accuracy: %g %%' % (accuracy_score * 100))




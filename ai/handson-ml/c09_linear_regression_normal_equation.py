import numpy as np
import tensorflow as tf
from california_housing_data import get_housing_data

housing = get_housing_data()
m,n = housing.data.shape
print('shape', m, n)

housing_data_plus_bias =  np.c_[np.ones((m, 1)), housing.data]

X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name='X')
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name='y')
XT = tf.transpose(X)
theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), y)
y_pred = tf.matmul(X, theta)
error = y_pred - y
mse = tf.reduce_mean(tf.square(error))

with tf.Session() as sess:
    theta_value = theta.eval()
    print(theta_value)

    #print('y', y.eval())
    #print('y_pred', y_pred.eval())
    print('error', error.eval())
    print('mse', mse.eval())

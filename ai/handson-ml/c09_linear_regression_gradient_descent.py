import numpy as np
import tensorflow as tf
from california_housing_data import get_scaled_housing_data

housing = get_scaled_housing_data()
m,n = housing.data.shape
print('shape', m, n)

housing_data_plus_bias =  np.c_[np.ones((m, 1)), housing.data]

n_epochs = 1000
learning_rate = 0.01

X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name='X')
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name='y')
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0), name='theta')
y_pred = tf.matmul(X, theta, name='predictions')
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name='mse')
gradients = tf.matmul(tf.transpose(X), error) / m
training_op = tf.assign(theta, theta - learning_rate * gradients)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print("Epoch", epoch, "MSE = ", mse.eval())
        sess.run(training_op)
    best_theta = theta.eval()
    print(best_theta)

    #print('y', y.eval())
    #print('y_pred', y_pred.eval())
    print('error', error.eval())
    print('mse', mse.eval())

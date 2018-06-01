import numpy as np
import tensorflow as tf
from california_housing_data import HousingData, HousingDataScaled

#housing = HousingData()
housing = HousingDataScaled()
m,n = housing.get_data_shape()
print('shape', m, n)

n_epochs = 1000
learning_rate = 0.01
batch_size = 100
n_batches =  int(np.ceil(m / batch_size))

X = tf.placeholder(tf.float32, shape=(None, n + 1), name='X')
y = tf.placeholder(tf.float32, shape=(None, 1), name='y')
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0), name='theta')
y_pred = tf.matmul(X, theta, name='predictions')
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name='mse')
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)

#init = tf.global_variables_initializer()
saver = tf.train.Saver() 

with tf.Session() as sess:
    saver.restore(sess, 'result/c09_linear_regression_mini_batch_final.ckpt')
    
    print('theta', theta.eval())

    X_all, y_all = housing.fetch_all()
    error_val, mse_val = sess.run([error, mse], feed_dict={X:X_all, y:y_all})
    print('error', error_val)
    print('mse', mse_val)

import numpy as np
import tensorflow as tf
from california_housing_data import HousingData, HousingDataScaled
from datetime import datetime

now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}".format(root_logdir, now)

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

mse_summary = tf.summary.scalar('MSE', mse)
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

init = tf.global_variables_initializer()
saver = tf.train.Saver() 

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            save_path = saver.save(sess, 'result/c09_linear_regression_mini_batch.ckpt')
        for batch_index in range(n_batches):
            X_batch, y_batch = housing.fetch_batch(batch_index, batch_size)
            if batch_index % 10 == 0:
                summary_str = mse_summary.eval(feed_dict={X:X_batch, y:y_batch})
                step = epoch * n_batches + batch_index
                file_writer.add_summary(summary_str, step)
            sess.run(training_op, feed_dict={X:X_batch, y:y_batch})
    
    file_writer.close()

    print('theta', theta.eval())
    save_path = saver.save(sess, 'result/c09_linear_regression_mini_batch_final.ckpt')
    print('save', save_path)

    X_all, y_all = housing.fetch_all()
    error_val, mse_val = sess.run([error, mse], feed_dict={X:X_all, y:y_all})
    print('error', error_val)
    print('mse', mse_val)

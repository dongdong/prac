import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf

a = tf.placeholder(tf.float32, shape=[3])
b = tf.constant([5, 5, 5], tf.float32)

c = a + b # short for tf.add(a, b)

writer = tf.summary.FileWriter('graphs/placeholder_1', tf.get_default_graph())

with tf.Session() as sess:
    print(sess.run(c, {a:[1, 2, 3]}))

writer.close()


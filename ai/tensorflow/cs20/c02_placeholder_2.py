import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf

a = tf.add(2, 5)
b = tf.multiply(a, 3)

writer = tf.summary.FileWriter('graphs/placeholder_2', tf.get_default_graph())

with tf.Session() as sess:
    print(sess.run(b))
    print(sess.run(b, feed_dict={a:15}))

writer.close()

#print(tf.get_default_graph().as_graph_def())

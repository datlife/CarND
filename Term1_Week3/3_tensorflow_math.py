# TensorFlow Documentation:
# Math Operations: https://www.tensorflow.org/api_docs/python/math_ops/

# Solution is available in the other "solution.py" tab
import tensorflow as tf

# TODO: Convert the following to TensorFlow:
x = tf.constant(10)
y = tf.constant(2)

division = tf.div(x, y)
z = tf.sub(division, tf.constant(1))

with tf.Session() as sess:
    output = sess.run(z)
    print(output)


import tensorflow as tf


def run():
    output = None
    x = tf.placeholder(tf.int32)
    y = tf.placeholder(tf.string)
    z = tf.placeholder(tf.float)

    with tf.Session() as sess:
        output = sess.run(x, feed_dict={x: 123, y: "hello", z: 3.12})

    return output

print(run())
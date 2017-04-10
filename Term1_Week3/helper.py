import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def print_epoch_stats(epoch_i, sess, last_features, last_labels):
    """
    Print epoch statistics.

    An epoch is a single forward and backward pass of the whole data set. This is used to increase accuracy of the model
    without requiring more data.
    :param epoch_i: a step forward and backward in a model
    :param sess: TensorFlow session
    :param last_features: last batch features in the model
    :param last_labels: last batch labels in the model
    :return: display to screen the statistic
    """
    pass


def weights(weight_name, n_features, n_labels):
    """
    Return TensorFlow weights
    :param weight_name: name of weight
    :param n_features: Number of features
    :param n_labels: Number of labels
    :return: TensorFlow weights
    """
    # TODO: Return weights
    return tf.get_variable(weight_name, [n_features, n_labels], initializer=tf.contrib.layers.xavier_initializer())


def biases(n_labels):
    """
    Return TensorFlow bias
    :param n_labels: Number of labels
    :return: TensorFlow bias
    """
    # TODO: Return biases
    b = tf.Variable(tf.zeros(n_labels))
    # b = tf.Variable(tf.truncated_normal((1, n_labels)))
    # b = tf.get_variable("b", [n_labels], initializer=tf.contrib.layers.xavier_initializer())

    return b


def linear(inputs, w, b):
    """
    Return linear function in TensorFlow
    :param inputs: TensorFlow input
    :param w: TensorFlow weights
    :param b: TensorFlow biases
    :return: TensorFlow linear function
    """
    # TODO: Linear Function (xW + b)
    l = tf.add(tf.matmul(inputs, w), b)
    return l


def batches(batch_size, features, labels):
    """
    Extract original data into mini-batches
    :param batch_size: size of each batch ( default = 64)
    :param features: number of features (input data)
    :param labels:  number of labels (output data)
    :return: array-like of mini batches
    """
    assert len(features) == len(labels)

    f_size = np.ceil(len(features)/batch_size)
    l_size = np.ceil(len(labels)/batch_size)

    feature_batch = np.array_split(features, f_size)
    label_batch   = np.array_split(labels, l_size)

    output = [[feature_batch[i], label_batch[i]] for i in range(len(feature_batch))]

    return output

    # assert len(features) == len(labels)
    # outout_batches = []
    #
    # sample_size = len(features)
    # for start_i in range(0, sample_size, batch_size):
    #     end_i = start_i + batch_size
    #     batch = [features[start_i:end_i], labels[start_i:end_i]]
    #     outout_batches.append(batch)
    #
    # return outout_batches


def mnist_features_labels(n_labels):
    """
    Gets the first <n> labels from the MNIST dataset
    :param n_labels: Number of labels to use
    :return: Tuple of feature list and label list
    """
    mnist_features = []
    mnist_labels = []

    mnist = input_data.read_data_sets("./MNIST_data", one_hot=True)

    # In order to make quizzes run faster, we're only looking at 10000 images
    for mnist_feature, mnist_label in zip(*mnist.train.next_batch(10000)):

        # Add features and labels if it's for the first <n>th labels
        if mnist_label[:n_labels].any():
            mnist_features.append(mnist_feature)
            mnist_labels.append(mnist_label[:n_labels])

    return mnist_features, mnist_labels


def variable_summaries(var):
    """
    Attach a lot of summaries to a Tensor (for TensorBoard visualization).
    :param var: A tensor variable (weight, biases)
    """
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
    tf.scalar_summary('mean', mean)
    with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.scalar_summary('stddev', stddev)
    tf.scalar_summary('max', tf.reduce_max(var))
    tf.scalar_summary('min', tf.reduce_min(var))
    tf.histogram_summary('histogram', var)

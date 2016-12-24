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



def weights(n_features, n_labels):
    """
    Return TensorFlow weights
    :param n_features: Number of features
    :param n_labels: Number of labels
    :return: TensorFlow weights
    """
    # TODO: Return weights
    w = tf.Variable(tf.truncated_normal((n_features, n_labels)))
    return w


def biases(n_labels):
    """
    Return TensorFlow bias
    :param n_labels: Number of labels
    :return: TensorFlow bias
    """
    # TODO: Return biases
    b = tf.Variable(tf.zeros(n_labels))
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

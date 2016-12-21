import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def mnist_features_labels(n_labels):
    """
    Gets the first <n> labels from the MNIST dataset
    :param n_labels: Number of labels to use
    :return: Tuple of feature list and label list
    """
    mnist_features = []
    mnist_labels = []

    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

    # In order to make quizzes run faster, we're only looking at 10000 images
    for mnist_feature, mnist_label in zip(*mnist.train.next_batch(10000)):

        # Add features and labels if it's for the first <n>th labels
        if mnist_label[:n_labels].any():
            mnist_features.append(mnist_feature)
            mnist_labels.append(mnist_label[:n_labels])

    return mnist_features, mnist_labels


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

# Number of features/inputs (28*28 image is 784 features)
n_features = 120
# number of outputs---- 1x5
n_labels = 5

# Features and labels
features = tf.placeholder(tf.float32)
labels   = tf.placeholder(tf.float32)

# Weights and biases
weights = weights(n_features, n_labels)
bias    = biases(n_labels)

# Linear Function xW + b
logits = linear(features, weights, bias)

# Training data
train_features, train_labels = mnist_features_labels(n_labels)

with tf.Session() as sess:
    # Initialize data
    sess.run(tf.initialize_all_variables())

    # Soft-max
    prediction = tf.nn.softmax(logits)

    # Cross entropy
    # Learn later - Calculate loss
    cross_entropy = -tf.reduce_sum(labels*tf.log(prediction), reduction_indices=1)
    loss = tf.reduce_mean(cross_entropy)

    # Set learning rate
    learning_rate = 0.08

    # Gradient Descent
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    # Run optimizer (back-prop?) and get the loss
    _, l = sess.run([optimizer, loss],
                    feed_dict={features: train_features, labels: train_labels})
# Print loss
print("Loss: {}, lr: {}".format(l, learning_rate))

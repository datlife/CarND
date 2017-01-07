import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
from tensorflow.contrib.layers import flatten

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", reshape=False)
X_train, Y_train = mnist.train.images, mnist.train.labels
X_validation, Y_validation = mnist.validation.images, mnist.validation.labels
X_test, Y_test = mnist.test.images, mnist.test.labels


# Display Image Specs:
print("\nImage shape: {}\n".format(X_train[0].shape))
print("Training set: {} samples.".format(len(X_train)))
print("Validation set: {} samples.".format(len(X_validation)))
print("Test set: {} samples.".format(len(X_test)))


# Pad images with 0s
X_train = np.pad(X_train, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
X_validation = np.pad(X_validation, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
X_test = np.pad(X_test, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')

print("Updated Image Shape: {}".format(X_train[0].shape))

# Pre-process Data - Shuffle Data
X_train, y_train = shuffle(X_train, Y_train)

# Build The CovNets
EPOCHS = 10
BATCH_SIZE = 128
LEARNING_RATE = 0.001


def conv_layer(x, W, b, strides=1):
    """
    Convolution Layer Wrapper
    :param data: input data
    :param W: sharing weight
    :param b: bias
    :param strides:
    :return: a convolution layer
    """
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='VALID')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def max_pool2d(x, k=2):
    return tf.nn.max_pool(
        x,
        ksize=[1, k, k, 1],
        strides=[1, k, k, 1],
        padding='SAME')


def LeNet(x):
    # Hyper-Parameters
    mu = 0
    sigma = 0.1

    weights = {
               'layer_1': tf.Variable(tf.truncated_normal([5, 5, 1, 6])),
               'layer_2': tf.Variable(tf.truncated_normal([5, 5, 6, 16])),
               'fc_1': tf.Variable(tf.truncated_normal([5*5*16, 120], mean=mu, stddev=sigma)),
               'fc_2': tf.Variable(tf.truncated_normal([120, 84], mean=mu, stddev=sigma)),
               'out': tf.Variable(tf.truncated_normal([84, 10], mean=mu, stddev=sigma))
              }
    biases = {
               'layer_1': tf.Variable(tf.zeros(6)),
               'layer_2': tf.Variable(tf.zeros(16)),
               'fc_1': tf.Variable(tf.zeros(120)),
               'fc_2': tf.Variable(tf.zeros(84)),
               'out': tf.Variable(tf.zeros(10))
    }
    # Layer 1: "Convolution" Layer:  Input 32x32x1 --> Output : 28x28x6 ---- 6 filter size = 5x5, stride = 1, pad = 0
    #            Activation        - ReLU
    #            Pooling           - MaxPooling --- 14x14x6
    layer_1 = conv_layer(x, weights['layer_1'], biases['layer_1'])
    layer_1 = max_pool2d(layer_1, k=2)

    # Layer 2: "Convolution" Layer:  Input 14x14x6 --> Output : 10x10x16 ---- 6 filter size = 5x5, stride = 1, pad = 0
    #            Activation       -  ReLU
    #            Pooling          -  MedianPooling -- 5x5x16
    layer_2 = conv_layer(layer_1, weights['layer_2'], biases['layer_2'])
    layer_2 = max_pool2d(layer_2, k=2)

    # Flatten Output : 5x5x16 --> 400
    flatten_layer = tf.contrib.layers.flatten(layer_2)

    # Layer 3: "Fully Connected" Layer: (Hidden Layer) Input: 400:  Output: 1x120
    #            Activation       -  ReLU
    fc1 = tf.add(tf.matmul(flatten_layer, weights['fc_1']), biases['fc_1'])

    # Layer 4: "Fully Connected" Layer: (Hidden Layer) Input: 120:  Output 84 outputs
    fc2 = tf.add(tf.matmul(fc1, weights['fc_2']), biases['fc_2'])

    # Layer 5 : Final Layer:                           Input 84  :  Output 10 output
    logits = tf.add(tf.matmul(fc2, weights['out']), biases['out'])

    return logits

# Features and Labels
x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, None)
one_hot_y = tf.one_hot(y, 10)

# Training Pipeline
logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_y)
loss = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
training_op = optimizer.minimize(loss)

# Model Evaluation
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()


def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy*len(batch_x))

    return total_accuracy/num_examples


# Train Model
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    num_examples = len(X_train)
    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, Y_train = shuffle(X_train, Y_train)
        print("EPOCH {} ...".format(i+1))
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], Y_train[offset:end]
            sess.run(training_op, feed_dict={x: batch_x, y: batch_y})

        validation_accuracy = evaluate(X_validation, Y_validation)
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()

    saver.save(sess, 'lenet')
    print("Model saved")
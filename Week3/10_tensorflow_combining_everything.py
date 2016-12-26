import tensorflow as tf
import numpy as np
from Week3.helper import weights, biases, linear, batches
from tensorflow.examples.tutorials.mnist import input_data

# Epochs - Numbers of forward and backward the whole data set [to improve accuracy]
epochs = 200
# Define learning rate
learning_rate = 0.0025
# Batch size
batch_size = 1024

n_features = 784
# Define number of possible output classes
n_classes = 10


# Import MNIST data set from Tensor Flow - Enabled one_hot to calculate Cross Entropy for Gradient Descent Optimizer
mnist = input_data.read_data_sets("./MNIST_data", one_hot=True)

# Get Training data set from mnist data
train_features = mnist.train.images
train_labels   = mnist.train.labels.astype(np.float32)

# Get Test Data set from mnist data  - DO NOT TOUCH until training is finished
test_features = mnist.test.images
test_labels   = mnist.test.labels.astype(np.float32)

# PRE-PROCESSING IMAGES HERE:
# 1. Zero-mean : (each_color_channel - 128)/128


# Since data is not fixed, using tf.placeholder() to store each data set
# Reference: Why 'None' is in exercise 9
features = tf.placeholder(np.float32, [None, n_features])
labels   = tf.placeholder(np.float32, [None, n_classes])

# Weights and biases for hidden layers - Types are tf.Variable()
weights = weights(n_features, n_classes)
biases  = biases(n_classes)

# Define Logistic Classifier
logits = linear(features, weights, biases)

# Hidden layers generate outputs stored in logit.
# Calculate Soft-max
soft_max = tf.nn.softmax(logits)

# Calculate cross - entropy using soft_max and one-hot encoding vector
cross_entropy = - tf.reduce_sum(labels * tf.log(soft_max))

# Get average cross entropy across all inputs to get a sense of how well models predict
# Loss is just average of cross entropy
loss = tf.reduce_mean(cross_entropy)

# Calculate Gradient Descent
# Could use Stochastic Gradient Descent for better performance. however, it requires to use smaller steps.
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# Create mini-batches from training data
batches = batches(batch_size, train_features, train_labels)

# Calculate accuracy
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


with tf.Session() as session:
    # Initialize all possible variables in Tensor Flow
    session.run(tf.initialize_all_variables())

    # Repeat training on same data
    for epoch in range(epochs):
        # Run training on all the mini-batches
        for train_feature, train_label in batches:
            _, l = session.run([optimizer, loss], feed_dict={features: train_feature, labels: train_label})

        # Test the model on test data set
        test_accuracy = session.run(
            accuracy,
            feed_dict={features: test_features, labels: test_labels})

        print('Epoch: {:<4} - Cost: {:<8.5} Valid Accuracy: {:<5.3}'.format(epoch, l, test_accuracy))

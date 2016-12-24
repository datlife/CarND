from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
from Week3.helper import weights, biases, linear, batches

n_input = 784   # MNIST data input (img shape : 28*28)
n_classes = 10  # MNIST total classes (objects)

# Import MNIST data
mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)

# The features are already scaled and the data is shuffled

# Training data set
train_features = mnist.train.images
train_labels   = mnist.train.labels.astype(np.float32)

# Test data set - DO NOT USE until training is completed
test_features  = mnist.test.images
test_labels    = mnist.test.labels.astype(np.float32)

# Weights and biases
weights = weights(n_input, n_classes)
biases  = biases(n_classes)

# define a Logistic Classifier


# What  does 'None' to here ?
# The 'None' dimension is a place holder for the batch size. At runtime, Tensor Flow will accept
# any batch size greater than 0 (feel like Dynamic Allocation in C++)
# Therefore, placeholder will be flexible to accept varying batch sizes ( 64, 128, 100)

# Features and labels
features = tf.placeholder(tf.float32, [None, n_input])
labels = tf.placeholder(tf.float32, [None, n_classes])

# Define original data set into mini-batches size of 128
batches = batches(128, train_features, train_labels)

print(batches)

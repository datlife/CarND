import tensorflow as tf
import numpy as np
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




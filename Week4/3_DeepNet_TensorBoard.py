import tensorflow as tf
import numpy as np
from Week3.helper import weights, biases, linear, batches, variable_summaries
from tensorflow.examples.tutorials.mnist import input_data

# /////////HYPER-PARAMETERS
epochs = 40
learning_rate = 0.001
batch_size = 128
# /////////////////////////

# path to saved model
save_file = 'model.ckpt'

n_features = 784
n_hidden = 256
n_classes = 10

# //////////////////////////////// IMPORT DATA SET //////////////////////////////////////////////
# Import MNIST data
mnist = input_data.read_data_sets("../Week3/MNIST_data", one_hot=True)
# Training Data Set
train_features = mnist.train.images
train_labels   = mnist.train.labels.astype(np.float32)
# Test Data Set  - DO NOT TOUCH until training is finished
test_features = mnist.test.images
test_labels   = mnist.test.labels.astype(np.float32)
# PRE-PROCESSING IMAGES HERE:
# 1. Zero-mean : (each_color_channel - 128)/128

# Mini-batches
batches = batches(batch_size, train_features, train_labels)

# ////////////////////////////// BUILD NEURAL NET ////////////////////////////////////////////
# Reference: Why 'None' is in exercise 9

features = tf.placeholder(np.float32, [None, n_features])
labels   = tf.placeholder(np.float32, [None, n_classes])

x_flat = tf.reshape(features, [-1, n_features])
# Weights and biases.
tf.variable_scope(tf.get_variable_scope(), reuse=False)

w = {'logit_layer': weights('logit_weights', n_features, n_hidden),
     'relu_layer': weights('reLU_weights', n_hidden, n_classes)}

b = {'logit_layer': biases(n_hidden),
     'relu_layer': biases(n_classes)}

# Logistic Classifier Layer
logits = linear(features, w['logit_layer'], b['logit_layer'])

# Rectified Linear Unit (ReLU) - Activation Function Layer
reLU = tf.nn.relu(logits)

# Output layer
output = tf.add(tf.matmul(reLU, w['relu_layer']), b['relu_layer'])

# ////////////////////////// OPTIMIZATION /////////////////////////////

# Soft-Max and Cross-Entropy to find distance
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(output, labels)
# Loss Values
loss = tf.reduce_mean(cross_entropy)

# Exponential Decay Learning Rate  - For ADAM-OPTIMIZER
epoch_step = tf.Variable(0)
exp_lr = tf.train.exponential_decay(learning_rate, epoch_step, len(batches), 0.95)

# AdamOptimizer - Required learning rate to be decayed
AdamOpt = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=epoch_step)

# Can I apply drop out here?

# Calculate accuracy
correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# ///////////////////////////VISUALIZATION/////////////////////////////////////
# TensorBoard - Debugging
# scalar_summary: values over time
# histogram_summary: value distribution from one particular layer.
recorder = tf.train.SummaryWriter('./logs/',graph=tf.get_default_graph())

with tf.Session() as session:
    tf.scalar_summary('loss', loss)
    tf.scalar_summary('accuracy', accuracy)
    # Merge all summaries and write them to ./tmp/mnist_logs (by_default)
    summary_op = tf.merge_all_summaries()
    # Initialize all possible variables in Tensor Flow
    session.run(tf.initialize_all_variables())

    # Repeat training on same data
    for epoch in range(epochs):
        # Run training on all the mini-batches
        for train_feature, train_label in batches:
            _, l, lr = session.run([AdamOpt, loss, exp_lr], feed_dict={features: train_feature, labels: train_label})

        # Test the model on test data set
        test_accuracy, log = session.run(
            [accuracy, summary_op],
            feed_dict={features: test_features, labels: test_labels})
        recorder.add_summary(log, epoch)

        print('Epoch: {:<4} - LR: {:9.5} - Cost: {:<8.5} Valid Accuracy: {:<5.4}'.format(epoch, lr, l, test_accuracy))

    # Save the model
    # saver.save(session, save_file)

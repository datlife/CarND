import tensorflow as tf
from Week3.helper import weights, biases, linear, mnist_features_labels
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

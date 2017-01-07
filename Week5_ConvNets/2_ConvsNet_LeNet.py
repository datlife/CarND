import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
from tensorflow.contrib.layers import flatten

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", reshape=False)
X_train, Y_train = mnist.train.images, mnist.train.labels
X_validation, Y_validation = mnist.validation.images, mnist.validation.labels
X_test, Y_test = mnist.test.images, mnist.test.labels


save_file = 'LeNet_model.ckpt'

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
KEEP_PROP = 0.5

def conv_layer(x, W, b, strides=1):
    """
    Convolution Layer Wrapper
    """
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='VALID')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def max_pool2d(x, k=2):
    return tf.nn.max_pool(
        x,
        ksize=[1, k, k, 1],
        strides=[1, k, k, 1],
        padding='VALID')


def weights(weight_name, size):
    """
    Return TensorFlow weights
    """
    return tf.get_variable(weight_name, size, initializer=tf.contrib.layers.xavier_initializer())


def evaluate(x_data, y_data, loss_op):
    num_of_examples = len(x_data)
    total_accuracy = 0
    session = tf.get_default_session()
    validation_loss = 0.0

    for offset in range(0, num_of_examples, BATCH_SIZE):
        bch_x, bch_y = x_data[offset:offset + BATCH_SIZE], y_data[offset:offset + BATCH_SIZE]
        accuracy, _loss= session.run([accuracy_operation, loss_op], feed_dict={x: bch_x, y: bch_y})
        total_accuracy += (accuracy*len(batch_x))
        validation_loss += _loss
    return total_accuracy/num_of_examples, validation_loss


def LeNet(x):

    tf.variable_scope(tf.get_variable_scope())
    w = {
               'layer_1': weights('layer_1', [5, 5, 1, 6]),
               'layer_2': weights('layer_2', [5, 5, 6, 16]),
               'fc_1': weights('fc_1', [5*5*16, 120]),
               'fc_2': weights('fc_2', [120, 84]),
               'out': weights('out', [84, 10])
              }
    biases = {
               'layer_1': tf.Variable(tf.zeros(6), name='bias_layer_1'),
               'layer_2': tf.Variable(tf.zeros(16), name='bias_layer_2'),
               'fc_1': tf.Variable(tf.zeros(120), name='bias_fc1'),
               'fc_2': tf.Variable(tf.zeros(84), name='bias_fc1'),
               'out': tf.Variable(tf.zeros(10), name='bias_logits')
    }

    # Layer 1: "Convolution" Layer:  Input 32x32x1 --> Output : 28x28x6 ---- 6 filter size = 5x5, stride = 1, pad = 0
    #            Activation        - ReLU
    #            Pooling           - MaxPooling --- 14x14x6
    layer_1 = conv_layer(x, w['layer_1'], biases['layer_1'])
    layer_1 = max_pool2d(layer_1, k=2)

    # Layer 2: "Convolution" Layer:  Input 14x14x6 --> Output : 10x10x16 ---- 6 filter size = 5x5, stride = 1, pad = 0
    #            Activation       -  ReLU
    #            Pooling          -  MedianPooling -- 5x5x16
    layer_2 = conv_layer(layer_1, w['layer_2'], biases['layer_2'])
    layer_2 = max_pool2d(layer_2, k=2)

    # Flatten Output : 5x5x16 --> 400
    flatten_layer = flatten(layer_2)

    # Layer 3: "Fully Connected" Layer: (Hidden Layer) Input: 400:  Output: 1x120
    #            Activation       -  ReLU
    fc1 = tf.add(tf.matmul(flatten_layer, w['fc_1']), biases['fc_1'])
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, keep_prob=0.5)
    # Layer 4: "Fully Connected" Layer: (Hidden Layer) Input: 120:  Output 84 outputs
    fc2 = tf.add(tf.matmul(fc1, w['fc_2']), biases['fc_2'])
    fc2 = tf.nn.relu(fc2)
    fc2 = tf.nn.dropout(fc2, keep_prob=0.5)

    # Layer 5 : Final Layer:                           Input 84  :  Output 10 output
    logits = tf.add(tf.matmul(fc2, w['out']), biases['out'])

    return logits


#
# ######## TRAINING PIPELINE #############
#

# Remove the previous weights and bias
tf.reset_default_graph()

# Features and Labels
x = tf.placeholder(tf.float32, (None, 32, 32, 1), name="features_holder")
y = tf.placeholder(tf.int32, (None), name="label_holder")
one_hot_y = tf.one_hot(y, 10)

# Load Network Architecture
logits = LeNet(x)
# Calculate Soft-max and Cross Entropy
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_y)
loss = tf.reduce_mean(cross_entropy)
# Apply AdamOptimizer
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
training_op = optimizer.minimize(loss)

# Apply L2 Regularization to avoid over-fitting
# Apply Batch Normalization

# Model Evaluation
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()


# ///////////////////////////VISUALIZATION/////////////////////////////////////
# TensorBoard - Debugging
# scalar_summary: values over time
# histogram_summary: value distribution from one particular layer.
recorder = tf.train.SummaryWriter('./logs/', graph=tf.get_default_graph())

# Train Model
with tf.Session() as sess:

    print("Start training...")
    try:
        saver.restore(sess, save_file)
        print("Restored Model Successfully.")
    except Exception as e:
        print(e)
        print("No model found...Start building a new one")
        sess.run(tf.initialize_all_variables())

    num_examples = len(X_train)
    for i in range(EPOCHS):
        X_train, Y_train = shuffle(X_train, Y_train)
        print("EPOCH {} ".format(i+1))
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], Y_train[offset:end]
            sess.run(training_op, feed_dict={x: batch_x, y: batch_y})

        validation_accuracy, val_loss = evaluate(X_validation, Y_validation, loss)
        print("Validation Loss {:<6.5f}  | Validation Accuracy = {:.3f}".format(val_loss, validation_accuracy))

        # tf.scalar_summary('loss', val_loss)
        # tf.scalar_summary('accuracy', validation_accuracy)
        # # Merge all summaries and write them to ./tmp/mnist_logs (by_default)
        # summary_op = tf.merge_all_summaries()
        # recorder.add_summary(summary_op, i+1)

    saver.save(sess, save_file)
    print("Train Model saved")
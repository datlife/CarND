"""
Check out the new network architecture and data set!

Notice that the weights and biases are generated randomly
"""

import numpy as np
from sklearn.datasets import load_boston
from sklearn.utils import shuffle, resample
from Week2.nnPy import *

# Load the data
data = load_boston()
X_ = data['data']
Y_ = data['target']

# Normalize data
X_ = (X_ - np.mean(X_, axis=0)) / np.std(X_, axis=0)

n_features = X_.shape[1]
n_hidden   = 10

W1_ = np.random.randn(n_features, n_hidden)
b1_ = np.zeros(n_hidden)
W2_ = np.random.randn(n_hidden, 1)
b2_ = np.zeros(1)

# Neural network
X, y = Input(), Input()
W1, b1 = Input(), Input()
W2, b2 = Input(), Input()

# Set up layers
l1 = Linear(X, W1, b1)
s1 = Sigmoid(l1)
l2 = Linear(s1, W2, b2)
cost = MSE(y, l2)

feed_dict = {
    X: X_,
    y: Y_,
    W1: W1_,
    b1: b1_,
    W2: W2_,
    b2: b2_,
}

# Number of steps
epochs = 1000

# Total number of samples
m = X_.shape[0]

# Default batch size, usually depends on GPU's memory
batch_size = 11
steps_per_epoch = m // batch_size

graph = topological_sort(feed_dict)
trainables = [W1, b1, W2, b2]

print('Total number of examples = {}'.format(m))

for i in range(epochs):
    loss = 0

    for j in range(steps_per_epoch):
        # Step 1: Randomly sample a batch of examples from original data
        X_batch, Y_batch = resample(X_, Y_, n_samples=batch_size)

        # Step 2: Update value of x and y inputs for neural network using new batch data
        X.value = X_batch
        y.value = Y_batch

        # Step 3 : Start training
        forward_and_backward(graph)

        # Step 4: Calculate Stochastic Gradient Descent
        sgd_update(trainables)

        # Update loss value
        loss += graph[-1].value

    print("Epoch: {}, Loss: {:.3f}".format(i+1, loss/steps_per_epoch))





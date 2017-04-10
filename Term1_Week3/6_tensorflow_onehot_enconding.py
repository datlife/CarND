"""
Using cross-entropy to calculate the Distance(Output, Expected_Values)
"""
import tensorflow as tf

# Cross entropy

# Distance(Soft_max values, Label) = - SUM of (Labels * Log (soft_max))

# Learn later - Calculate loss
cross_entropy = -tf.reduce_sum(labels * tf.log(prediction), reduction_indices=1)
loss = tf.reduce_mean(cross_entropy)

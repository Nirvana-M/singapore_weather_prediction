"""
Modified VGG suitable for CIFAR-10
"""

import logging

import tensorflow as tf
from tensorflow.contrib import rnn

logger = logging.getLogger(__name__)

# parameter(s) for cnn
pad = 'SAME'

# parameter(s) for lstm
n_steps = 1
n_hidden = 128
n_classes = 11*9
# Define weights for lstm
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}

def cnn_feature_extractor(x1):

    with tf.variable_scope('conv1'):
        x1 = tf.layers.conv2d(x1, 8, (3, 3), padding=pad, activation=tf.nn.relu, name="conv1_1")
        x1 = tf.layers.conv2d(x1, 8, (3, 3), padding=pad, activation=tf.nn.relu, name="conv1_2")
        x1 = tf.layers.max_pooling2d(x1, (2, 2), (2, 2), name="pool1")


    with tf.variable_scope('conv2'):
        x1 = tf.layers.conv2d(x1, 16, (3, 3), padding=pad, activation=tf.nn.relu, name="conv2_1")
        x1 = tf.layers.conv2d(x1, 16, (3, 3), padding=pad, activation=tf.nn.relu, name="conv2_2")
        x1 = tf.layers.max_pooling2d(x1, (2, 2), (2, 2), name="pool2")

    return x1

def build(x):
    assert str(x.get_shape()) == "(?, 30, 11, 9)"
    logger.info("Building VGG model")

    log = lambda x: logger.info("\t{}\t{}".format(x.get_shape(), x.name))

    STEP_SIZE = x.get_shape()[1]

    x = tf.transpose(x,[1,0,2,3])

    y = x[0]
    y = tf.reshape(y, [-1, 11, 9, 1])
    y = cnn_feature_extractor(y)
    y = tf.reshape(y, [-1, 1, 64])

    x = y


    log(x)
    with tf.variable_scope('lstm'):
        # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
        x = tf.unstack(x, n_steps, 1)
        # Define a lstm cell with tensorflow
        lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
        # Get lstm cell output
        outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
        # Linear activation, using rnn inner loop last output
        return tf.matmul(outputs[-1], weights['out']) + biases['out']

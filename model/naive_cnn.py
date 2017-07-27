"""
Modified VGG suitable for singapore_weather_prediction
"""

import logging

import tensorflow as tf

logger = logging.getLogger(__name__)

pad = 'SAME'

def build(x):
    assert str(x.get_shape()) == "(?, 11, 9, 1)"
    logger.info("Building VGG model")

    log = lambda x: logger.info("\t{}\t{}".format(x.get_shape(), x.name))

    log(x)
    with tf.variable_scope('conv1'):
        x = tf.layers.conv2d(x, 8, (3, 3), padding=pad, activation=tf.nn.relu, name="conv1_1")
        x = tf.layers.conv2d(x, 8, (3, 3), padding=pad, activation=tf.nn.relu, name="conv1_2")
        x = tf.layers.max_pooling2d(x, (2, 2), (2, 2), name="pool1")

    log(x)
    with tf.variable_scope('conv2'):
        x = tf.layers.conv2d(x, 16, (3, 3), padding=pad, activation=tf.nn.relu, name="conv2_1")
        x = tf.layers.conv2d(x, 16, (3, 3), padding=pad, activation=tf.nn.relu, name="conv2_2")
        x = tf.layers.max_pooling2d(x, (2, 2), (2, 2), name="pool2")

    log(x)
    return tf.reshape(x,[-1,64])

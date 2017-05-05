import os
import time
import sys

import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

import utils

def variable_summaries(var, name):
  """Attach a lot of summaries to a Tensor."""
  with tf.name_scope('summaries'):
    varscope = tf.get_variable_scope().name
    name = "%s/%s" % (varscope, name)
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean/' + name, mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
    tf.summary.scalar('sttdev/' + name, stddev)
    tf.summary.scalar('max/' + name, tf.reduce_max(var))
    tf.summary.scalar('min/' + name, tf.reduce_min(var))
    tf.summary.histogram(name, var)

def _variable(name, initializer, cpu=True):
    """Helper to create variable:
    Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable
    Returns:
    Variable Tensor
    """
    if cpu:
        with tf.device('/cpu:0'):
            var = tf.get_variable(name, initializer=initializer)
    else:
        var = tf.get_variable(name, initializer=initializer)
    return var

def _variable_with_weight_decay(name, wd, init, cpu=True):
    """Helper to create an initialized Variable with weight decay.
    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.
    Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.
    Returns:
    Variable Tensor
    """
    var = _variable(name, init, cpu=cpu)

    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID', name='conv')

def _maybe_pad_x(x, padding, is_training):
    if padding == 0:
       x_pad = x
    elif padding > 0:
        x_pad = tf.cond(is_training, lambda: x,
                        lambda: utils.replicate_padding(x, padding))
    else:
        raise ValueError("Padding value %i should be greater than or equal to 1" % padding)
    return x_pad

def inference(images, input_depth, num_filters, filter_sizes, wd=0.0,
              keep_prob=1.0, is_training=True, weights=None, biases=None,
              multigpu=True):
    if isinstance(is_training, bool):
       is_training = tf.constant(is_training)

    if isinstance(wd, float) or isinstance(wd, int):
        wd = [float(wd)] * len(num_filters)

    if isinstance(keep_prob, float) or isinstance(keep_prob, int):
        keep_prob = tf.constant(keep_prob, shape=[], name='keep_prob')

    if weights is not None:
        assert len(weights) == len(biases) == len(num_filters) == len(filter_sizes)

    w_conv, b_conv, h_conv = [], [], []
    for i, k in enumerate(filter_sizes[:-1]):
        if i == 0:
            c = input_depth
            x_in = images
        else:
            c = num_filters[i-1]
            x_in = h_conv

        with tf.variable_scope("hidden%i" % i) as scope:
            pad_amt = (k-1)/2
            x_padded = _maybe_pad_x(x_in, pad_amt, is_training)
            w_shape = [k, k, c, num_filters[i]]
            b_shape = [num_filters[i]]

            if weights is not None:
                init = tf.constant(weights[i])
                init_bias = tf.constant(biases[i])
            else:
                init=tf.truncated_normal(w_shape, stddev=0.001)
                init_bias = tf.zeros(b_shape)

            w_conv = _variable_with_weight_decay('W', wd=wd[i], init=init, cpu=multigpu)
            b_conv = _variable('bias', init_bias, cpu=multigpu)
            conv = conv2d(x_padded, w_conv)

            h = tf.nn.bias_add(conv, b_conv)
            h_conv = tf.nn.relu(h, name="hconv")

    with tf.variable_scope('output') as scope:
        noise_shape = [tf.shape(h_conv)[0], 1, 1, tf.shape(h_conv)[3]]
        x_in = tf.nn.dropout(h_conv, keep_prob, noise_shape=noise_shape)
        pad_amt = (filter_sizes[-1]-1)/2
        x_pad = _maybe_pad_x(x_in, pad_amt, is_training)

        w_shape = [filter_sizes[-1], filter_sizes[-1], num_filters[-2],
                            num_filters[-1]]
        if weights is not None:
            init = tf.constant(weights[-1])
            init_bias = tf.constant(biases[-1])
        else:
            init=tf.truncated_normal(w_shape, stddev=0.001)
            init_bias = tf.zeros([num_filters[-1]])

        w_conv3 = _variable_with_weight_decay('W', wd=wd[-1], init=init, cpu=multigpu)
        b_conv3 = _variable('bias', init_bias, cpu=multigpu)
        conv3 = tf.add(conv2d(x_pad, w_conv3), b_conv3, name='prediction')
    return conv3

def loss(predictions, labels, alpha=1.):
    err = tf.square(predictions - labels)
    err_filled = utils.fill_na(err, 0)
    finite_count = tf.reduce_sum(tf.cast(tf.is_finite(err), tf.float32))
    mse = alpha * tf.reduce_sum(err_filled) / finite_count
    #mse = tf.reduce_mean(err) / 2
    tf.add_to_collection('losses', mse/2.)
    return mse/2.

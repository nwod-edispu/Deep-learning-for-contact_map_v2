#!/usr/bin/env python

import tensorflow as tf

# database files
tf.flags.DEFINE_string(
    'train_file', 'train.pkl',
    'Directory where checkpoints and event logs are written to.')

tf.compat.v1.app.flags.DEFINE_string(
    'valid_file', 'valid.pkl',
    'Directory where checkpoints and event logs are written to.')

tf.compat.v1.app.flags.DEFINE_string(
    'test_file', 'test.pkl',
    'Directory where checkpoints and event logs are written to.')

# dir paths
tf.compat.v1.app.flags.DEFINE_string(
    'train_dir', './output/residual_network/',
    'Directory where checkpoints and event logs are written to.')

tf.compat.v1.app.flags.DEFINE_string(
    'data_dir', './data/',
    'Directory of database.')

# network building params
tf.compat.v1.app.flags.DEFINE_integer(
    'filter_size_1d', 17,
    'filter size for 1D conv.')

tf.compat.v1.app.flags.DEFINE_integer(
    'filter_size_2d', 3,
    'filter size for 2D conv.')

tf.compat.v1.app.flags.DEFINE_integer(
    'block_num_1d', 1,
    'num of residual block for 1D conv.')

tf.compat.v1.app.flags.DEFINE_integer(
    'block_num_2d', 20,
    'num of residual block for 2D conv.')

# net training params
tf.compat.v1.app.flags.DEFINE_integer(
    'max_iters', 100000,
    'maximum iteration times')

# restore model
tf.compat.v1.app.flags.DEFINE_bool(
    'restore_previous_if_exists', True,
    'restore models trained previous')

FLAGS = tf.compat.v1.app.flags.FLAGS

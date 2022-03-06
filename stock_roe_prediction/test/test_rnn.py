"""
======================
# -*-coding: utf8-*-
# @Author  : louiss007
# @Time    : 22-3-6 下午9:37
# @FileName: test_rnn.py
# @Email   : quant_master2000@163.com
======================
"""

import tensorflow as tf
import numpy as np

# hyperparameters
hidden_size = 4
X_data = np.array([
        # steps   1st     2nd       3rd
        [[1.0, 2], [7, 8], [13, 14]],   # first batch
        [[3, 4], [9, 10], [15, 16]],    # second batch
        [[5, 6], [11, 12], [17, 18]]    # third batch
])      # shape: [batch_size, n_steps, n_inputs]
# parameters
n_steps = X_data.shape[1]
n_inputs = X_data.shape[2]

# rnn model
X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])

output, state = tf.nn.dynamic_rnn(tf.nn.rnn_cell.BasicRNNCell(hidden_size), X, dtype=tf.float32)
# initializer the variables
init = tf.global_variables_initializer()

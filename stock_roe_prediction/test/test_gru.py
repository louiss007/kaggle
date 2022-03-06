"""
======================
# -*-coding: utf8-*-
# @Author  : louiss007
# @Time    : 22-3-6 下午6:23
# @FileName: test_gru.py
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
])  # shape: [batch_size, n_steps, n_inputs]

# parameters
n_steps = X_data.shape[1]
n_inputs = X_data.shape[2]
print('n_steps: ', n_steps)
print('n_inputs', n_inputs)

# rnn model
X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])

output, state = tf.nn.dynamic_rnn(tf.nn.rnn_cell.GRUCell(hidden_size), X, dtype=tf.float32)
# initializer the variables
init = tf.global_variables_initializer()

# train
with tf.Session() as sess:
    sess.run(init)
    feed_dict = {X: X_data}
    output = sess.run(output, feed_dict=feed_dict)
    state = sess.run(state, feed_dict=feed_dict)
    print('output: \n', output[-1])     # 所有sequence token的hidden state
    print('output shape [batch_size, n_steps, n_neurons]: ', np.shape(output))
    print('state: \n', state)           # 最后的hidden state
    print('state shape [n_layers, batch_size, n_neurons]: ', np.shape(state))

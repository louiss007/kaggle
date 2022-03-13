"""
======================
# -*-coding: utf8-*-
# @Author  : louiss007
# @Time    : 22-1-23 下午9:00
# @FileName: bilstm_model.py
# @Email   : quant_master2000@163.com
======================
"""
from lstm_model import lstm_model
import tensorflow as tf


class bilstm_model(lstm_model):

    def __init__(self, model_para, out_para, task_type=None):
        lstm_model.__init__(self, model_para, out_para, task_type)
        self.time_steps = None
        self.model_path = '{mp}/nn/bilstm/bilstm'.format(mp=out_para.get('model_path'))
        self.init_net()
        if self.task_type == 'regression':
            self.loss, self.train_op = self.build_model()
        else:
            self.loss, self.train_op, self.accuracy = self.build_model()

    def init_net(self):
        self.time_steps = 28
        self.X = tf.placeholder(tf.float32, [None, self.time_steps, self.layers[0]])
        self.Y = tf.placeholder(tf.float32, [None, self.num_classes])
        self.weights = {
            'out': tf.Variable(tf.random_normal([2 * self.layers[1], self.num_classes]))
        }
        self.biases = {
            'out': tf.Variable(tf.random_normal([self.num_classes]))
        }

    def neural_network(self, x):
        # Prepare data shape to match `rnn` function requirements
        # Current data input shape: (batch_size, timesteps, n_input)
        # Required shape: 'timesteps' tensors list of shape (batch_size, num_input)

        # Unstack to get a list of 'timesteps' tensors of shape (batch_size, num_input)
        x = tf.unstack(x, self.time_steps, 1)

        # Define lstm cells with tensorflow
        # Forward direction cell
        lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.layers[1], forget_bias=1.0, reuse=tf.AUTO_REUSE)
        # Backward direction cell
        lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.layers[1], forget_bias=1.0, reuse=tf.AUTO_REUSE)

        # Get lstm cell output
        outputs, _, _ = tf.nn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype=tf.float32)
        # Linear activation, using rnn inner loop last output
        out_layer = tf.matmul(outputs[-1], self.weights['out']) + self.biases['out']
        return out_layer

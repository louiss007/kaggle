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
            self.loss, self.train_op = self.bulid_model()
        else:
            self.loss, self.train_op, self.accuracy = self.bulid_model()

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

    # def bulid_model(self):
    #     """
    #     构建模型，损失函数，优化器，学习算子等
    #     :return:
    #     """
    #     y_hat = self.neural_network(self.X)
    #     if self.task_type is None or self.task_type == 'classification':
    #         self.out = tf.nn.softmax(logits=y_hat)
    #         loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_hat, labels=self.Y))
    #         optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
    #         train_op = optimizer.minimize(loss, global_step=self.global_step)
    #         corr_pred = tf.equal(tf.argmax(self.out, 1), tf.argmax(self.Y, 1))
    #         accuracy = tf.reduce_mean(tf.cast(corr_pred, tf.float32))
    #         return loss, train_op, accuracy
    #
    #     if self.task_type == 'regression':
    #         loss = tf.reduce_mean(tf.square(y_hat - self.Y))
    #         # loss = tf.reduce_mean(tf.square(y_hat - self.Y), keep_dims=False)
    #         optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
    #         train_op = optimizer.minimize(loss, global_step=self.global_step)
    #         self.out = y_hat
    #         return loss, train_op
    #
    # def parse_tfrecord(self, tfrecord):
    #     print("=============bilstm model=================")
    #     example = tf.parse_single_example(tfrecord, features={
    #         'image': tf.FixedLenFeature([], tf.string),
    #         'label': tf.FixedLenFeature([], tf.string),
    #         'num1': tf.FixedLenFeature([], tf.float32),
    #         'num2': tf.FixedLenFeature([], tf.int64)
    #     })
    #     image = tf.decode_raw(example['image'], tf.float32)
    #     label = tf.decode_raw(example['label'], tf.float32)
    #     image = tf.reshape(image, shape=[self.time_steps, self.layers[0]])
    #     # image = tf.reshape(image, shape=[self.layers[0]])
    #     label = tf.reshape(label, shape=[self.num_classes])
    #     return image, label

"""
======================
# -*-coding: utf8-*-
# @Author  : louiss007
# @Time    : 22-1-24 下午11:26
# @FileName: cnn_model.py
# @Email   : quant_master2000@163.com
======================
"""
from nn_model import nn_model
import tensorflow as tf
import numpy as np


class cnn_model(nn_model):

    def __init__(self, model_para, out_para, task_type=None):
        """
        模型构建初始化
        :param model_para: 模型学习时的超参数
        :param out_para: 模型输出的参数
        :param task_type: 分类/回归/排序等
        """
        nn_model.__init__(self, model_para, out_para, task_type)
        self.height = 28
        self.width = 28
        self.channels = 1
        self.model_path = '{mp}/nn/cnn/cnn'.format(mp=out_para.get('model_path'))
        self.init_net()
        if self.task_type == 'regression':
            self.loss, self.train_op = self.bulid_model()
        else:
            self.loss, self.train_op, self.accuracy = self.bulid_model()

    def init_net(self):
        self.X = tf.placeholder(tf.float32, [None, self.layers[0]])
        self.Y = tf.placeholder(tf.float32, [None, self.num_classes])
        # Store layers weight & bias
        self.weights = {
            # 5x5 conv, 1 input, 32 outputs
            'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
            # 5x5 conv, 32 inputs, 64 outputs
            'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
            # fully connected, 7*7*64 inputs, 1024 outputs
            'wd1': tf.Variable(tf.random_normal([7 * 7 * 64, 1024])),
            # 1024 inputs, 10 outputs (class prediction)
            'out': tf.Variable(tf.random_normal([1024, self.num_classes]))
        }

        self.biases = {
            'bc1': tf.Variable(tf.random_normal([32])),
            'bc2': tf.Variable(tf.random_normal([64])),
            'bd1': tf.Variable(tf.random_normal([1024])),
            'out': tf.Variable(tf.random_normal([self.num_classes]))
        }
        # if len(self.layers) != 1:
        #     for i in range(1, len(self.layers)):
        #         init_method = np.sqrt(2.0 / (self.layers[i - 1] + self.layers[i]))
        #         self.weights['h' + str(i)] = tf.Variable(np.random.normal(
        #             loc=0, scale=init_method, size=(self.layers[i - 1], self.layers[i])),
        #             dtype=np.float32)
        #         self.biases['b' + str(i)] = tf.Variable(
        #             np.random.normal(loc=0, scale=init_method, size=(1, self.layers[i])),
        #             dtype=np.float32)
        #     self.weights['out'] = tf.Variable(tf.random_normal([self.layers[-1], self.num_classes]))
        #     self.biases['out'] = tf.Variable(tf.random_normal([self.num_classes]))

    def neural_network(self, x):
        x = tf.reshape(x, shape=[-1, 28, 28, 1])

        # Convolution Layer
        conv1 = self.conv2d(x, self.weights['wc1'], self.biases['bc1'])
        # Max Pooling (down-sampling)
        conv1 = self.maxpool2d(conv1, k=2)

        # Convolution Layer
        conv2 = self.conv2d(conv1, self.weights['wc2'], self.biases['bc2'])
        # Max Pooling (down-sampling)
        conv2 = self.maxpool2d(conv2, k=2)

        # Fully connected layer
        # Reshape conv2 output to fit fully connected layer input
        fc1 = tf.reshape(conv2, [-1, self.weights['wd1'].get_shape().as_list()[0]])
        fc1 = tf.add(tf.matmul(fc1, self.weights['wd1']), self.biases['bd1'])
        fc1 = tf.nn.relu(fc1)
        # Apply Dropout
        fc1 = tf.nn.dropout(fc1, self.dropout)

        # Output, class prediction
        out = tf.add(tf.matmul(fc1, self.weights['out']), self.biases['out'])
        return out

    # def bulid_model(self):
    #     """
    #     构建模型，损失函数，优化器，学习算子等
    #     :return:
    #     """
    #     y_hat = self.neural_network(self.X)
    #     if self.task_type is None or self.task_type == 'classification':
    #         self.out = tf.nn.softmax(logits=y_hat)
    #         loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_hat, labels=self.Y))
    #         optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
    #         train_op = optimizer.minimize(loss, global_step=self.global_step)
    #         corr_pred = tf.equal(tf.argmax(self.out, 1), tf.argmax(self.Y, 1))
    #         accuracy = tf.reduce_mean(tf.cast(corr_pred, tf.float32))
    #         return loss, train_op, accuracy
    #
    #     if self.task_type == 'regression':
    #         loss = tf.reduce_mean(tf.square(y_hat - self.Y))
    #         # loss = tf.reduce_mean(tf.square(y_hat - self.Y), keep_dims=False)
    #         optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
    #         train_op = optimizer.minimize(loss, global_step=self.global_step)
    #         self.out = y_hat
    #         return loss, train_op

    def conv2d(self, x, W, b, strides=1):
        # Conv2D wrapper, with bias and relu activation
        x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)

    def maxpool2d(self, x, k=2):
        # MaxPool2D wrapper
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

    def parse_tfrecord(self, tfrecord):
        example = tf.parse_single_example(tfrecord, features={
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.string),
            'num1': tf.FixedLenFeature([], tf.float32),
            'num2': tf.FixedLenFeature([], tf.int64)
        })
        image = tf.decode_raw(example['image'], tf.float32)
        label = tf.decode_raw(example['label'], tf.float32)
        # num1 = example['num1']
        # num2 = example['num2']
        image = tf.reshape(image, shape=[self.height, self.width, self.channels])
        image = tf.reshape(image, shape=[self.layers[0]])
        label = tf.reshape(label, shape=[self.num_classes])
        return image, label

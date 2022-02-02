"""
======================
# -*-coding: utf8-*-
# @Author  : louiss007
# @Time    : 22-1-24 下午11:28
# @FileName: nn_model.py
# @Email   : quant_master2000@163.com
======================
"""
import tensorflow as tf
import numpy as np
import pandas as pd


class nn_model:

    def __init__(self, model_para, out_para, task_type=None):
        self.task_type = task_type
        self.X = None
        self.Y = None
        self.weights = None
        self.biases = None
        self.learning_rate = model_para.get('learning_rate')
        self.epoch = model_para.get('epoch')
        self.batch_size = model_para.get('batch_size')
        self.num_classes = model_para.get('num_classes')
        self.layers = model_para.get('layers')
        self.global_step = tf.Variable(0, trainable=False)
        self.out = None
        self.init_net()
        self.loss, _, self.accuracy = self.bulid_model()

    def init_net(self):
        self.X = tf.placeholder('float', [None, self.layers[0]])
        self.Y = tf.placeholder('float', [None, self.num_classes])
        if len(self.layers) != 1:
            for i in range(1, len(self.layers)):
                init_method = np.sqrt(2.0 / (self.layers[i - 1] + self.layers[i]))
                self.weights['h' + str(i)] = tf.Variable(
                    np.random.normal(loc=0, scale=init_method, size=(self.layers[i - 1], self.layers[i])),
                    dtype=np.float32)
                self.biases['b' + str(i)] = tf.Variable(
                    np.random.normal(loc=0, scale=init_method, size=(1, self.layers[i])),
                    dtype=np.float32)
            self.weights['out'] = tf.Variable(tf.random_normal([self.layers[i], self.num_classes]))
            self.biases['out'] = tf.Variable(tf.random_normal([self.num_classes]))

    def neural_network(self, x):
        layer_1 = tf.add(tf.matmul(x, self.weights['h1']), self.biases['b1'])
        layer_2 = tf.add(tf.matmul(layer_1, self.weights['h2']), self.biases['b2'])
        out_layer = tf.matmul(layer_2, self.weights['out']) + self.biases['out']
        return out_layer

    def bulid_model(self):
        y_hat = self.neural_network(self.X)
        self.out = tf.nn.softmax(logits=y_hat)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_hat, labels=self.Y))
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        train_op = optimizer.minimize(loss)
        if self.task_type is None or self.task_type == 'classification':
            corr_pred = tf.equal(tf.argmax(self.out, 1), tf.argmax(self.Y, 1))
            accuracy = tf.reduce_mean(tf.cast(corr_pred, tf.float32))
            return loss, train_op, accuracy

        if self.task_type == 'regression':
            loss = tf.reduce_mean(tf.square(y_hat - self.Y))
            self.out = y_hat
            return loss, train_op

    def fit(self, sess, batch_x, batch_y):
        loss, acc, global_step = sess.run(
            [self.loss, self.accuracy, self.global_step], feed_dict={
                self.X: batch_x,
                self.Y: batch_y
            })
        return loss, acc, global_step

    def batch_fit(self, X, Y):
        with tf.Session() as sess:
            pass

    def predict(self, sess, x, y):
        result = sess.run([self.out], feed_dict={
            self.X: x,
            self.Y: y
        })
        return result

    def eval(self):
        pass

    def save_model(self, sess, path):
        saver = tf.train.Saver()
        saver.save(sess, save_path=path)

    def restore_model(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, save_path=path)

    def make_one_batch(self, batch_size, i, train: pd.DataFrame, train_size):
        """
        input is dataframe format
        :param batch_size:
        :param i:
        :param train:
        :param train_size:
        :return: dataframe format
        """
        if (i+1)*batch_size > train_size:
            batch_data = train.iloc[i * batch_size: train_size, :]
        else:
            batch_data = train.iloc[i*batch_size: (i+1)*batch_size, :]
        return batch_data

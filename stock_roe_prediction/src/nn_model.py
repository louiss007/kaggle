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


class nn_model:

    def __init__(self, para_map, task_type=None):
        self.task_type = task_type
        self.X = None
        self.Y = None
        self.weights = None
        self.biases = None
        self.layers = para_map.get('layers')
        self.epoch = para_map.get('epoch')
        self.batch_size = para_map.get('batch_size')
        self.num_classes = para_map.get('num_classes')
        self.global_step = tf.Variable(0, trainable=False)
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

    def neural_network(self, x):
        layer_1 = tf.add(tf.matmul(x, self.weights['h1']), self.biases['b1'])
        layer_2 = tf.add(tf.matmul(layer_1, self.weights['h2']), self.biases['b2'])
        out_layer = tf.matmul(layer_2, self.weights['out']) + self.biases['out']
        return out_layer

    def bulid_model(self):
        out = self.neural_network(self.X)
        y = tf.nn.softmax(logits=out)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=self.Y))
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        train_op = optimizer.minimize(loss)
        corr_pred = tf.equal(tf.argmax(y, 1), tf.argmax(self.Y, 1))
        accuracy = tf.reduce_mean(tf.cast(corr_pred, tf.float32))
        if self.task_type == 'regression':
            #TODO
            pass
        return loss, train_op, accuracy

    def fit(self, sess, batch_x, batch_y):
        loss, acc, global_step = sess.run(
            [self.loss, self.accuracy, self.global_step], feed_dict={
            self.X: batch_x,
            self.Y: batch_y
        })
        return loss, acc, global_step

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

    def make_one_batch(self, index, step):
        pass

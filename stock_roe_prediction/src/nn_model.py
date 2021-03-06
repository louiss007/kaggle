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

    def __init__(self, model_para, out_para, task_type=None):
        """
        模型构建初始化
        :param model_para: 模型学习时的超参数
        :param out_para: 模型输出的参数
        :param task_type: 分类/回归/排序等
        """
        self.task_type = task_type
        self.X = None
        self.Y = None
        self.feat_cols = model_para.get('feat_cols')
        self.target = model_para.get('target')
        self.weights = {}
        self.biases = {}
        self.dropout = model_para.get('dropout')
        self.learning_rate = model_para.get('learning_rate')
        self.epoch = model_para.get('epoch')
        # self.n_step = model_para.get('n_step')
        self.train_sample_size = model_para.get('train_sample_size')
        self.test_sample_size = model_para.get('test_sample_size')
        self.batch_size = model_para.get('batch_size')
        self.n_step = self.train_sample_size//self.batch_size + 1
        self.num_classes = model_para.get('num_classes')
        if self.task_type == 'regression':
            self.num_classes = 1
        self.layers = model_para.get('layers')
        self.global_step = tf.Variable(0, trainable=False)
        self.model_path = '{mp}/nn/fnn/fnn'.format(mp=out_para.get('model_path'))
        self.out = None
        self.init_net()
        if self.task_type == 'regression':
            self.loss, self.train_op = self.build_model()
        else:
            self.loss, self.train_op, self.accuracy = self.build_model()

    def init_net(self):
        """
        与下面的网络结构相对应，是下面网络结构中的权重矩阵定义与数据输入定义，
        修改下面网络结构时，此函数中对应的权重矩阵也要对应的修改
        :return:
        """
        self.X = tf.placeholder(tf.float32, [None, self.layers[0]])
        self.Y = tf.placeholder(tf.float32, [None, self.num_classes])
        if len(self.layers) != 1:
            for i in range(1, len(self.layers)):
                init_method = np.sqrt(2.0 / (self.layers[i - 1] + self.layers[i]))
                self.weights['h' + str(i)] = tf.Variable(np.random.normal(
                    loc=0, scale=init_method, size=(self.layers[i - 1], self.layers[i])),
                    dtype=np.float32)
                self.biases['b' + str(i)] = tf.Variable(
                    np.random.normal(loc=0, scale=init_method, size=(1, self.layers[i])),
                    dtype=np.float32)
            self.weights['out'] = tf.Variable(tf.random_normal([self.layers[-1], self.num_classes]))
            self.biases['out'] = tf.Variable(tf.random_normal([self.num_classes]))

    def neural_network(self, x):
        """
        网络结构，正向传播输出，可以替换为其他任意构造的网络结构
        :param x: input tensor
        :return: y_hat
        """
        layer_1 = tf.add(tf.matmul(x, self.weights['h1']), self.biases['b1'])
        layer_2 = tf.add(tf.matmul(layer_1, self.weights['h2']), self.biases['b2'])
        out_layer = tf.matmul(layer_2, self.weights['out']) + self.biases['out']
        return out_layer

    def build_model(self):
        """
        构建模型，损失函数，优化器，学习算子等
        :return:
        """
        y_hat = self.neural_network(self.X)
        if self.task_type is None or self.task_type == 'classification':
            self.out = tf.nn.softmax(logits=y_hat)
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_hat, labels=self.Y))
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            train_op = optimizer.minimize(loss, global_step=self.global_step)
            corr_pred = tf.equal(tf.argmax(self.out, 1), tf.argmax(self.Y, 1))
            accuracy = tf.reduce_mean(tf.cast(corr_pred, tf.float32))
            return loss, train_op, accuracy

        if self.task_type == 'regression':
            loss = tf.reduce_mean(tf.square(y_hat - self.Y))
            # loss = tf.reduce_mean(tf.square(y_hat - self.Y), keep_dims=False)
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            train_op = optimizer.minimize(loss, global_step=self.global_step)
            self.out = y_hat
            return loss, train_op

    def fit(self, sess, batch_x, batch_y):
        """
        learn model, 训练模型
        :param sess:
        :param batch_x: 特征
        :param batch_y: 标签
        :return:
        """
        if self.task_type is None or self.task_type == 'classification':
            loss, acc, _, step = sess.run(
                [self.loss, self.accuracy, self.train_op, self.global_step], feed_dict={
                    self.X: batch_x,
                    self.Y: batch_y
                })
            return loss, acc, step
        if self.task_type == 'regression':
            loss, _, step = sess.run(
                [self.loss, self.train_op, self.global_step], feed_dict={
                    self.X: batch_x,
                    self.Y: batch_y
                })
            return loss, step

    def predict(self, sess, x, y):
        """
        模型预测
        :param sess:
        :param x: 特征
        :param y: y is None
        :return: y_hat
        """
        result = sess.run([self.out], feed_dict={
            self.X: x,
            self.Y: y
        })
        return result

    def eval(self):
        pass

    def save_model(self, sess, path):
        saver = tf.train.Saver()
        saver.save(sess, save_path=path + '.ckpt', global_step=self.global_step)

    def restore_model(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, save_path=path)

    def parse_tfrecord(self, tfrecord):
        features = {}
        for col in self.feat_cols:
            features.setdefault(col, tf.FixedLenFeature([], tf.float32))
        features.setdefault(self.target, tf.FixedLenFeature([], tf.float32))
        example = tf.parse_single_example(tfrecord, features=features)
        x = [example[col] for col in self.feat_cols]
        y = [example[self.target]]
        return x, y

    def make_one_batch(self, tfrecord_files):
        dataset = tf.data.TFRecordDataset(tfrecord_files)
        dataset = dataset.map(self.parse_tfrecord).batch(self.batch_size)
        iterator = dataset.make_one_shot_iterator()
        batch_x, batch_y = iterator.get_next()
        return batch_x, batch_y

    def make_batch(self, tfrecord_files):
        dataset = tf.data.TFRecordDataset(tfrecord_files)
        dataset = dataset.map(self.parse_tfrecord).batch(256)
        iterator = dataset.make_one_shot_iterator()
        batch_x, batch_y = iterator.get_next()
        return batch_x, batch_y

    @staticmethod
    def check_shape(t):
        print('==========================')
        print(t.op.name, ' ', t.get_shape().as_list())


if __name__ == '__main__':
    pass

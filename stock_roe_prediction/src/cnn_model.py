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


class cnn_model(nn_model):

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
        self.learning_rate = model_para.get('learning_rate')
        self.epoch = model_para.get('epoch')
        # self.n_step = model_para.get('n_step')
        self.train_sample_size = model_para.get('train_sample_size')
        self.test_sample_size = model_para.get('test_sample_size')
        self.batch_size = model_para.get('batch_size')
        self.n_step = self.train_sample_size // self.batch_size + 1
        self.num_classes = model_para.get('num_classes')
        if self.task_type == 'regression':
            self.num_classes = 1
        self.layers = model_para.get('layers')
        self.global_step = tf.Variable(0, trainable=False)
        self.model_path = '{mp}/nn/fnn/fnn'.format(mp=out_para.get('model_path'))
        self.out = None
        self.init_net()
        if self.task_type == 'regression':
            self.loss, self.train_op = self.bulid_model()
        else:
            self.loss, self.train_op, self.accuracy = self.bulid_model()

    def init_net(self):
        pass

    def neural_network(self, x):
        pass
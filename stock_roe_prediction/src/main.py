"""
==========================
# -*- coding: utf8 -*-
# @Author   : louiss007
# @Time     : 2022/1/21 22:40
# @FileName : main.py
# @Email    : quant_master2000@163.com
==========================
"""
from utils.general_utils import *
from xgb_model import xgb_model
from lgb_model import lgb_model
from nn_model import nn_model
import sys
import json
"""
程序运行入口
"""


def get_tfrecord_files(data_path):
    train_tfrecord_files = []
    test_tfrecord_files = []
    for i in range(1, 15):
        file = '%s/train_%02d.tfrecord' % (data_path, i)
        train_tfrecord_files.append(file)
    last_file = '%s/train_%02d.tfrecord' % (data_path, 16)
    test_file = '%s/train_%02d.tfrecord' % (data_path, 15)
    train_tfrecord_files.append(last_file)
    test_tfrecord_files.append(test_file)
    return train_tfrecord_files, test_tfrecord_files


def get_feat_cols(feat_json_file):
    fi = open(feat_json_file, 'r')
    feat_map = json.loads(fi.read())
    feat_cols = feat_map.keys()
    return feat_cols


def nn_run(train_files, test_files, model_para, out_para, task_type, is_train=True):
    model = nn_model(model_para, out_para, task_type)
    # if not tf.gfile.Exists(model.model_path):
    #     tf.gfile.MakeDirs(model.model_path)

    with tf.Session() as sess:
        variables_initner = tf.global_variables_initializer()
        tables_initner = tf.tables_initializer()
        sess.run(variables_initner)
        sess.run(tables_initner)
        if is_train:
            for epoch in range(model.epoch):
                _x, _y = model.make_one_batch(train_files)  # must in epoch loop, not in step loop
                for step in range(model.n_step):
                    batch_x, batch_y = sess.run([_x, _y])
                    if task_type is None or task_type == 'classification':
                        loss, acc, global_step = model.fit(sess, batch_x, batch_y)
                        if global_step % 2000 == 0:
                            print('==========loss:{0}, acc:{1}, epoch:{2}, global step:{3}======'
                                  .format(loss, acc, epoch, global_step))
                            model.save_model(sess, model.model_path)
                    if task_type == 'regression':
                        loss, global_step = model.fit(sess, batch_x, batch_y)
                        if global_step % 2000 == 0:
                            print('==========loss:{0}, epoch:{1}, global step:{2}======'
                                  .format(loss, epoch, global_step))
                            model.save_model(sess, model.model_path)
        else:
            test_n_step = model.test_sample_size // model.batch_size + 1
            for _ in range(test_n_step):
                batch_x, batch_y = model.make_one_batch(test_files)
                _x, _y = sess.run([batch_x, batch_y])
                result = model.predict(sess, _x, _y)


def run(model_type, task_type=None):
    """
    模型训练程序运行入口
    :param model_type: 具体机器学习模型或者深度学习模型
    :param task_type: 分类、回归或者排序
    :return: null
    """

    if model_type == 'xgb':
        # conf = sys.argv[1]
        conf = '../conf/xgb_conf.yaml'
        para_map = read_conf(conf)
        in_para = para_map.get('in')
        # model_para = para_map.get('binary-class')
        model_para = para_map.get('regression')
        out_para = para_map.get('out')
        data, feats, target = get_data(in_para)
        xgb = xgb_model(model_para, out_para)
        xgb.fit_delta(data, feats, target)

    if model_type == 'lgb':
        # conf = sys.argv[1]
        conf = '../conf/lgb_conf.yaml'
        para_map = read_conf(conf)
        in_para = para_map.get('in')
        # model_para = para_map.get('binary-class')
        model_para = para_map.get('regression')
        out_para = para_map.get('out')
        data, feats, target = get_data(in_para)
        lgb = lgb_model(model_para, out_para)
        lgb.fit_delta(data, feats, target)

    if model_type == 'fnn':
        conf = '../conf/nn_conf.yaml'
        para_map = read_conf(conf)
        in_para = para_map.get('in')
        nn_para = para_map.get(model_type)
        out_para = para_map.get('out')
        train_sample_size = in_para.get('train_sample_size')
        train_sample_size -= 200000
        test_sample_size = in_para.get('test_sample_size')
        # n_step = train_sample_size // nn_para.get('batch_size') + 1
        target = 'target'
        # ds, feats, target = get_dataset(in_para, nn_para.get('batch_size'))
        json_file = '{dp}/feat.json'.format(dp=in_para.get('data_path'))
        feats = get_feat_cols(json_file)
        nn_para.setdefault('feat_cols', feats)
        nn_para.setdefault('target', target)
        nn_para.setdefault('train_sample_size', train_sample_size)
        nn_para.setdefault('test_sample_size', test_sample_size)
        train_files, test_files = get_tfrecord_files(in_para.get('data_path'))
        nn_run(train_files, test_files, nn_para, out_para, task_type)


if __name__ == '__main__':
    model_type = 'fnn'
    # model_type = 'xgb'
    # model_type = 'lgb'
    run(model_type, 'regression')

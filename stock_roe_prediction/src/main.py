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
from cnn_model import cnn_model
from alexnet_model import alexnet_model
from rnn_model import rnn_model
from lstm_model import lstm_model
from bilstm_model import bilstm_model
from gru_model import gru_model
from gan_model import gan_model
from dcgan_model import dcgan_model
# from dcgan_model_v2 import dcgan_model
import time
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


def nn_run_for_unsupervised(train_files, test_files, model_para, model, task_type, is_train=True):
    # model = cnn_model(model_para, out_para, task_type)
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
                    feed_dict = model.make_one_batch_for_g(batch_x)
                    g_loss, d_loss, global_step = model.fit(sess, feed_dict)
                    if global_step % model_para.get('display_step') == 0:
                        print('==========g_loss:{0}, d_loss:{1}, epoch:{2}, global step:{3}======'
                              .format(g_loss, d_loss, epoch, global_step))
                        model.save_model(sess, model.model_path)
            model.show_fake_image(sess)

                # print('===========validation start===========')
                # test_x, test_y = model.make_batch(test_files)
                # t_x, t_y = sess.run([test_x, test_y])
                # loss, acc, _ = model.fit(sess, t_x, t_y)
                # print('==========test loss:{0}, test acc:{1}, epoch:{2}======'
                #       .format(loss, acc, epoch))

        else:
            test_n_step = model.test_sample_size // model.batch_size + 1
            for _ in range(test_n_step):
                batch_x, batch_y = model.make_one_batch(test_files)
                _x, _y = sess.run([batch_x, batch_y])
                result = model.predict(sess, _x, _y)


def nn_run(train_files, test_files, model_para, model, task_type, is_train=True):
    # model = cnn_model(model_para, out_para, task_type)
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
                        if global_step % model_para.get('display_step') == 0:
                            print('==========train loss:{0}, train acc:{1}, epoch:{2}, global step:{3}======'
                                  .format(loss, acc, epoch, global_step))
                            model.save_model(sess, model.model_path)
                    if task_type == 'regression':
                        loss, global_step = model.fit(sess, batch_x, batch_y)
                        if global_step % model_para.get('display_step') == 0:
                            print('==========train loss:{0}, epoch:{1}, global step:{2}======'
                                  .format(loss, epoch, global_step))
                            model.save_model(sess, model.model_path)

                print('===========validation start===========')
                test_x, test_y = model.make_batch(test_files)
                t_x, t_y = sess.run([test_x, test_y])
                if task_type is None or task_type == 'classification':
                    loss, acc, _ = model.fit(sess, t_x, t_y)
                    print('==========test loss:{0}, test acc:{1}, epoch:{2}======'
                          .format(loss, acc, epoch))
                        # model.save_model(sess, model.model_path)
                if task_type == 'regression':
                    loss, _ = model.fit(sess, t_x, t_y)
                    print('==========test loss:{0}, epoch:{1}======'
                          .format(loss, epoch))

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
        train_sample_size -= 200000     # only one total dataset
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
        model = nn_model(nn_para, out_para, task_type)
        nn_run(train_files, test_files, nn_para, model, task_type)

    if model_type == 'cnn':
        conf = '../conf/nn_conf.yaml'
        para_map = read_conf(conf)
        in_para = para_map.get('cnn_in')
        nn_para = para_map.get(model_type)
        out_para = para_map.get('out')
        train_sample_size = in_para.get('train_sample_size')
        # train_sample_size -= 200000
        test_sample_size = in_para.get('test_sample_size')
        # n_step = train_sample_size // nn_para.get('batch_size') + 1
        target = 'target'
        # ds, feats, target = get_dataset(in_para, nn_para.get('batch_size'))
        # json_file = '{dp}/feat.json'.format(dp=in_para.get('data_path'))
        # feats = get_feat_cols(json_file)
        feats = None
        nn_para.setdefault('feat_cols', feats)
        nn_para.setdefault('target', target)
        nn_para.setdefault('train_sample_size', train_sample_size)
        nn_para.setdefault('test_sample_size', test_sample_size)
        # train_files, test_files = get_tfrecord_files(in_para.get('data_path'))
        train_files = '{dp}/train.tfrecord'.format(dp=in_para.get('data_path'))
        test_files = '{dp}/test.tfrecord'.format(dp=in_para.get('data_path'))
        model = cnn_model(nn_para, out_para, task_type)
        nn_run(train_files, test_files, nn_para, model, task_type)

    if model_type == 'alexnet':
        conf = '../conf/nn_conf.yaml'
        para_map = read_conf(conf)
        in_para = para_map.get('alexnet_in')
        nn_para = para_map.get(model_type)
        out_para = para_map.get('out')
        train_sample_size = in_para.get('train_sample_size')
        # train_sample_size -= 200000
        test_sample_size = in_para.get('test_sample_size')
        # n_step = train_sample_size // nn_para.get('batch_size') + 1
        target = 'target'
        # ds, feats, target = get_dataset(in_para, nn_para.get('batch_size'))
        # json_file = '{dp}/feat.json'.format(dp=in_para.get('data_path'))
        # feats = get_feat_cols(json_file)
        feats = None
        nn_para.setdefault('feat_cols', feats)
        nn_para.setdefault('target', target)
        nn_para.setdefault('train_sample_size', train_sample_size)
        nn_para.setdefault('test_sample_size', test_sample_size)
        # train_files, test_files = get_tfrecord_files(in_para.get('data_path'))
        train_files = '{dp}/train.tfrecord'.format(dp=in_para.get('data_path'))
        test_files = '{dp}/test.tfrecord'.format(dp=in_para.get('data_path'))
        model = alexnet_model(nn_para, out_para, task_type)
        nn_run(train_files, test_files, nn_para, model, task_type)

    if model_type == 'rnn':
        conf = '../conf/nn_conf.yaml'
        para_map = read_conf(conf)
        in_para = para_map.get('rnn_in')
        nn_para = para_map.get(model_type)
        out_para = para_map.get('out')
        train_sample_size = in_para.get('train_sample_size')
        # train_sample_size -= 200000
        test_sample_size = in_para.get('test_sample_size')
        # n_step = train_sample_size // nn_para.get('batch_size') + 1
        target = 'target'
        # ds, feats, target = get_dataset(in_para, nn_para.get('batch_size'))
        # json_file = '{dp}/feat.json'.format(dp=in_para.get('data_path'))
        # feats = get_feat_cols(json_file)
        feats = None
        nn_para.setdefault('feat_cols', feats)
        nn_para.setdefault('target', target)
        nn_para.setdefault('train_sample_size', train_sample_size)
        nn_para.setdefault('test_sample_size', test_sample_size)
        # train_files, test_files = get_tfrecord_files(in_para.get('data_path'))
        train_files = '{dp}/train.tfrecord'.format(dp=in_para.get('data_path'))
        test_files = '{dp}/test.tfrecord'.format(dp=in_para.get('data_path'))
        model = rnn_model(nn_para, out_para, task_type)
        nn_run(train_files, test_files, nn_para, model, task_type)

    if model_type == 'lstm':
        conf = '../conf/nn_conf.yaml'
        para_map = read_conf(conf)
        in_para = para_map.get('lstm_in')
        nn_para = para_map.get(model_type)
        out_para = para_map.get('out')
        train_sample_size = in_para.get('train_sample_size')
        # train_sample_size -= 200000
        test_sample_size = in_para.get('test_sample_size')
        # n_step = train_sample_size // nn_para.get('batch_size') + 1
        target = 'target'
        # ds, feats, target = get_dataset(in_para, nn_para.get('batch_size'))
        # json_file = '{dp}/feat.json'.format(dp=in_para.get('data_path'))
        # feats = get_feat_cols(json_file)
        feats = None
        nn_para.setdefault('feat_cols', feats)
        nn_para.setdefault('target', target)
        nn_para.setdefault('train_sample_size', train_sample_size)
        nn_para.setdefault('test_sample_size', test_sample_size)
        # train_files, test_files = get_tfrecord_files(in_para.get('data_path'))
        train_files = '{dp}/train.tfrecord'.format(dp=in_para.get('data_path'))
        test_files = '{dp}/test.tfrecord'.format(dp=in_para.get('data_path'))
        model = lstm_model(nn_para, out_para, task_type)
        nn_run(train_files, test_files, nn_para, model, task_type)

    if model_type == 'bilstm':
        conf = '../conf/nn_conf.yaml'
        para_map = read_conf(conf)
        in_para = para_map.get('bilstm_in')
        nn_para = para_map.get(model_type)
        out_para = para_map.get('out')
        train_sample_size = in_para.get('train_sample_size')
        # train_sample_size -= 200000
        test_sample_size = in_para.get('test_sample_size')
        # n_step = train_sample_size // nn_para.get('batch_size') + 1
        target = 'target'
        # ds, feats, target = get_dataset(in_para, nn_para.get('batch_size'))
        # json_file = '{dp}/feat.json'.format(dp=in_para.get('data_path'))
        # feats = get_feat_cols(json_file)
        feats = None
        nn_para.setdefault('feat_cols', feats)
        nn_para.setdefault('target', target)
        nn_para.setdefault('train_sample_size', train_sample_size)
        nn_para.setdefault('test_sample_size', test_sample_size)
        # train_files, test_files = get_tfrecord_files(in_para.get('data_path'))
        train_files = '{dp}/train.tfrecord'.format(dp=in_para.get('data_path'))
        test_files = '{dp}/test.tfrecord'.format(dp=in_para.get('data_path'))
        model = bilstm_model(nn_para, out_para, task_type)
        nn_run(train_files, test_files, nn_para, model, task_type)

    if model_type == 'gru':
        conf = '../conf/nn_conf.yaml'
        para_map = read_conf(conf)
        in_para = para_map.get('gru_in')
        nn_para = para_map.get(model_type)
        out_para = para_map.get('out')
        train_sample_size = in_para.get('train_sample_size')
        # train_sample_size -= 200000
        test_sample_size = in_para.get('test_sample_size')
        # n_step = train_sample_size // nn_para.get('batch_size') + 1
        target = 'target'
        # ds, feats, target = get_dataset(in_para, nn_para.get('batch_size'))
        # json_file = '{dp}/feat.json'.format(dp=in_para.get('data_path'))
        # feats = get_feat_cols(json_file)
        feats = None
        nn_para.setdefault('feat_cols', feats)
        nn_para.setdefault('target', target)
        nn_para.setdefault('train_sample_size', train_sample_size)
        nn_para.setdefault('test_sample_size', test_sample_size)
        # train_files, test_files = get_tfrecord_files(in_para.get('data_path'))
        train_files = '{dp}/train.tfrecord'.format(dp=in_para.get('data_path'))
        test_files = '{dp}/test.tfrecord'.format(dp=in_para.get('data_path'))
        model = gru_model(nn_para, out_para, task_type)
        nn_run(train_files, test_files, nn_para, model, task_type)

    if model_type == 'gan':
        conf = '../conf/nn_conf.yaml'
        para_map = read_conf(conf)
        in_para = para_map.get('gan_in')
        nn_para = para_map.get(model_type)
        out_para = para_map.get('out')
        train_sample_size = in_para.get('train_sample_size')
        # train_sample_size -= 200000
        test_sample_size = in_para.get('test_sample_size')
        # n_step = train_sample_size // nn_para.get('batch_size') + 1
        target = 'target'
        # ds, feats, target = get_dataset(in_para, nn_para.get('batch_size'))
        # json_file = '{dp}/feat.json'.format(dp=in_para.get('data_path'))
        # feats = get_feat_cols(json_file)
        feats = None
        nn_para.setdefault('feat_cols', feats)
        nn_para.setdefault('target', target)
        nn_para.setdefault('train_sample_size', train_sample_size)
        nn_para.setdefault('test_sample_size', test_sample_size)
        # train_files, test_files = get_tfrecord_files(in_para.get('data_path'))
        train_files = '{dp}/train.tfrecord'.format(dp=in_para.get('data_path'))
        test_files = '{dp}/test.tfrecord'.format(dp=in_para.get('data_path'))
        model = gan_model(nn_para, out_para, task_type)
        nn_run_for_unsupervised(train_files, test_files, nn_para, model, task_type)

    if model_type == 'dcgan':
        conf = '../conf/nn_conf.yaml'
        para_map = read_conf(conf)
        in_para = para_map.get('dcgan_in')
        nn_para = para_map.get(model_type)
        out_para = para_map.get('out')
        train_sample_size = in_para.get('train_sample_size')
        # train_sample_size -= 200000
        test_sample_size = in_para.get('test_sample_size')
        # n_step = train_sample_size // nn_para.get('batch_size') + 1
        target = 'target'
        # ds, feats, target = get_dataset(in_para, nn_para.get('batch_size'))
        # json_file = '{dp}/feat.json'.format(dp=in_para.get('data_path'))
        # feats = get_feat_cols(json_file)
        feats = None
        nn_para.setdefault('feat_cols', feats)
        nn_para.setdefault('target', target)
        nn_para.setdefault('train_sample_size', train_sample_size)
        nn_para.setdefault('test_sample_size', test_sample_size)
        # train_files, test_files = get_tfrecord_files(in_para.get('data_path'))
        train_files = '{dp}/train.tfrecord'.format(dp=in_para.get('data_path'))
        test_files = '{dp}/test.tfrecord'.format(dp=in_para.get('data_path'))
        model = dcgan_model(nn_para, out_para, task_type)
        nn_run_for_unsupervised(train_files, test_files, nn_para, model, task_type)


if __name__ == '__main__':
    # model_type = 'fnn'
    # model_type = 'xgb'
    # model_type = 'lgb'
    # model_type = 'cnn'
    # model_type = 'alexnet'
    # model_type = 'rnn'
    # model_type = 'lstm'
    # model_type = 'bilstm'
    # model_type = 'gru'
    # model_type = 'gan'
    model_type = 'dcgan'
    start = time.time()
    # run(model_type, 'regression')
    run(model_type, 'classification')
    end = time.time()
    print('elapsed time: %d' % (end-start))


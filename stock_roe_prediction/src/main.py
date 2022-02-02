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

"""
程序运行入口
"""


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
        # model_para = para_map.get('binary-class')
        model_para = para_map.get('regression')
        in_para = para_map.get('in')
        out_para = para_map.get('out')
        xgb = xgb_model(model_para, out_para)
        data, feats, target = get_data(in_para)
        xgb.fit_delta(data, feats, target)

    if model_type == 'lgb':
        # conf = sys.argv[1]
        conf = '../conf/lgb_conf.yaml'
        para_map = read_conf(conf)
        # model_para = para_map.get('binary-class')
        model_para = para_map.get('regression')
        in_para = para_map.get('in')
        out_para = para_map.get('out')
        lgb = lgb_model(model_para, out_para)
        data, feats, target = get_data(in_para)
        lgb.fit_delta(data, feats, target)

    if model_type == 'fnn':
        conf = '../conf/nn_conf.yaml'
        para_map = read_conf(conf)
        nn_para = para_map.get(model_type)
        in_para = para_map.get('in')
        out_para = para_map.get('out')
        fnn = nn_model(nn_para, out_para)
        data, feats, target = get_data(in_para)

        input_size = len(feats)
        nn_para['layers'][0] = input_size
        fnn.fit(nn_para, out_para)


if __name__ == '__main__':
    model_type = 'fnn'
    run(model_type)

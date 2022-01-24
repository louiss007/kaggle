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
import sys

"""
程序运行入口
"""


def run(model_type):

    if model_type == 'xgb':
        # conf = sys.argv[1]
        conf = '../conf/xgb_conf.yaml'
        para_map = read_conf(conf)
        # model_para = para_map.get('binary-class')
        model_para = para_map.get('regression')
        data_path = para_map.get('data')
        model_path = para_map.get('model')
        xgb = xgb_model(model_para, model_path)
        xgb.fit_delta(data_path)

    if model_type == 'lgb':
        # conf = sys.argv[1]
        conf = '../conf/lgb_conf.yaml'
        para_map = read_conf(conf)
        # model_para = para_map.get('binary-class')
        model_para = para_map.get('regression')
        data_path = para_map.get('data')
        model_path = para_map.get('model')
        lgb = lgb_model(model_para, model_path)
        lgb.fit_delta(data_path)


if __name__ == '__main__':
    model_type = 'lgb'
    run(model_type)

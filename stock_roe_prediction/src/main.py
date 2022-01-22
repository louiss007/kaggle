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
import sys

"""
程序运行入口
"""


def run():
    xgb = xgb_model
    conf = sys.argv[1]
    para_map = read_conf(conf)
    model_para = para_map.get('binary-class')
    data_path = para_map.get('data')
    model_path = para_map.get('model')
    xgb.fit(model_para, data_path, model_path)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("usage: python %s conf" % __file__)
        sys.exit()
    run()

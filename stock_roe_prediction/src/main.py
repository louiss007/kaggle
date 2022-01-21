"""
==========================
# -*- coding: utf8 -*-
# @Author   : louiss007
# @Time     : 2022/1/21 22:40
# @FileName : main.py
# @Email    : quant_master2000@163.com
==========================
"""
from utils.general_utils import load_data
from xgb_model import xgb_model

"""
程序运行入口
"""


def run():
    load_data()
    xgb = xgb_model
    xgb.fit()


if __name__ == '__main__':
    run()


"""
==========================
# -*- coding: utf8 -*-
# @Author   : louiss007
# @Time     : 2022/1/21 22:27
# @FileName : general_utils.py
# @Email    : quant_master2000@163.com
==========================
"""

import pandas as pd
import codecs
import yaml


def load_data():
    pass


def read_conf(infile):
    fi = codecs.open(infile, 'r', 'utf-8')
    data = fi.read()
    fi.close()
    para_map = yaml.load(data, yaml.BaseLoader)
    return para_map




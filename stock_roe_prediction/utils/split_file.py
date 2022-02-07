"""
======================
# -*-coding: utf8-*-
# @Author  : louiss007
# @Time    : 22-2-7 下午8:52
# @FileName: split_file.py
# @Email   : quant_master2000@163.com
======================
"""

import os


def shell_cmd(cmd):
    ret = os.system(cmd)
    if not ret:
        print('{0} succeed!'.format(cmd))
    else:
        print('{0} failed!'.format(cmd))


def split_file(path):
    source_path = '{0}/jpg'.format(path)
    # files = os.listdir(source_path)
    for i in range(7, 18):
        target_path = '{0}/jpg/{1}'.format(path, i)
        start = (i-1)*80+1
        end = i*80
        for k in range(start, end+1):
            image = '%s/image_%04d.jpg' % (source_path, k)
            cmd = 'mv {0} {1}/'.format(image, target_path)
            # print(cmd)
            shell_cmd(cmd)


if __name__ == '__main__':
    source_path = '/home/louiss007/Downloads/pic_data'
    split_file(source_path)


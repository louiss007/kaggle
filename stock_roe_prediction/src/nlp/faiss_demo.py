"""
======================
# -*-coding: utf8-*-
# @Author  : louiss007
# @Time    : 22-4-4 下午11:40
# @FileName: faiss_demo.py
# @Email   : quant_master2000@163.com
======================
"""
import faiss
import pickle
import numpy as np
import time


if __name__ == '__main__':
    x = np.random.random((200, 100)).astype('float32')
    # print(x)
    d = 100
    num = 5
    niter = 3
    kmeans = faiss.Kmeans(d, num, niter=niter, gpu=True)
    kmeans.train(x)
    D, I = kmeans.index.search(x, 1)
    print(I)

"""
==========================
# -*- coding: utf8 -*-
# @Author   : louiss007
# @Time     : 2022/1/21 22:26
# @FileName : xgb_model.py
# @Email    : quant_master2000@163.com
==========================
"""
import xgboost as xgb
import matplotlib.pyplot as plt
from xgboost import plot_importance
from sklearn.metrics import *
from utils.general_utils import load_data
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import train_test_split

import pandas as pd
# import graphviz


class xgb_model:

    def __init__(self, param, out_para):
        self.param = param
        self.out_para = out_para

    def fit(self, data, feats, target):
        """
        compare train loss with test loss,
        judge for overfitting or underfitting
        :param data:
        :param feats:
        :param target:
        :return:
        """
        X = data[feats]
        y = data[target]
        groups = data['time_id']
        kfold = GroupKFold(n_splits=10)
        num_round = 20
        for fold, (t_ind, v_ind) in enumerate(kfold.split(X, y, groups)):
            print(f"==================fold: {fold}==================")
            X_t, y_t = X.iloc[t_ind], y.iloc[t_ind]
            X_v, y_v = X.iloc[v_ind], y.iloc[v_ind]
            dtrain = xgb.DMatrix(X_t, label=y_t)
            deval = xgb.DMatrix(X_v, label=y_v)
            watchlist = [(dtrain, 'train'), (deval, 'val')]
            bst = xgb.train(self.param, dtrain, num_round, verbose_eval=1, evals=watchlist)
            model_bin = "{mp}/xgb_model_{f}.bin".format(mp=self.out_para.get('model_path'), f=fold)
            # xgb.to_graphviz(bst, num_trees=0)
            bst.save_model(model_bin)
            # dump_path = model_map.get('dump_path')
            # model2txt(model_bin, dump_path)
            # feature_importance(bst)
        # return bst

    def fit_delta(self, data, feats, target):
        """
        not online learning mode, because data contains all samples
        :param data: dataframe format
        :param feats:
        :param target:
        :return:
        """
        train, val = train_test_split(data, test_size=0.2, random_state=1)
        train_size = len(train)
        X_v = val[feats]
        y_v = val[target]
        deval = xgb.DMatrix(X_v, label=y_v)
        batch_size = 100000
        steps = train_size // batch_size
        num_round = 20
        model_bin = "{mp}/xgb_model.bin".format(mp=self.out_para.get('model_path'))
        for i in range(steps+1):
            print(f"===============step:{i}=================")
            batch_data = self.make_one_batch(batch_size, i, train, train_size)
            X_t = batch_data[feats]
            y_t = batch_data[target]
            dtrain = xgb.DMatrix(X_t, label=y_t)
            watchlist = [(dtrain, 'train'), (deval, 'val')]
            try:
                bst = xgb.train(self.param, dtrain, num_round, verbose_eval=1, evals=watchlist, xgb_model=model_bin)
                bst.save_model(model_bin)
            except:
                bst = xgb.train(self.param, dtrain, num_round, verbose_eval=1, evals=watchlist)
                bst.save_model(model_bin)

    def make_one_batch(self, batch_size, i, train: pd.DataFrame, train_size):
        if (i+1)*batch_size > train_size:
            batch_data = train.iloc[i * batch_size: train_size, :]
        else:
            batch_data = train.iloc[i*batch_size: (i+1)*batch_size, :]
        return batch_data

    def eval(self, bst, dtest):
        tlabel = dtest.get_label()
        predictions = bst.predict(dtest)
        plabel = [round(val) for val in predictions]
        auc = roc_auc_score(tlabel, plabel)
        print("auc: %.3f" % auc)

    def predict(self, model_bin, dtest):
        pass

    def batch_predict(self, model_bin, dtest):
        bst = self.load_model(model_bin)
        tlabel = dtest.get_label()
        predictions = bst.predict(dtest)
        plabel = [round(val) for val in predictions]
        auc = roc_auc_score(tlabel, plabel)
        print("auc: %.3f" % auc)

    def graph_visual(self, model_bin):
        bst = self.load_model(model_bin)
        xgb.plot_tree(bst, num_trees=0, rankdir='LR')
        plt.show()
        plt.savefig('out.png', dpi=300)
        # xgb.to_graphviz(bst, num_trees=0)

    def feature_importance(self, model):
        fig, ax = plt.subplots(figsize=(15, 15))
        plot_importance(model,
                        height=0.5,
                        ax=ax,
                        max_num_features=64)
        plt.show()

    def model2txt(self, model_bin, dump_path):
        bst = self.load_model(model_bin)
        model_txt = "{dp}/xgb_model.txt".format(dp=dump_path)
        # feature_map = "{dp}/xgb_feature_map.txt".format(dp=dump_path)
        bst.dump_model(model_txt)
        # bst.dump_model(model_txt, feature_map)

    def save_model(self, bst, path):
        bst.save_model(path)

    def load_model(self, model):
        bst = xgb.Booster({'nthread': 4})
        bst.load_model(model)
        return bst

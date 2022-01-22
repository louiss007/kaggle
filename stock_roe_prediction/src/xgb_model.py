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
import graphviz


class xgb_model:

    def __init__(self):
        pass

    def fit(self, param, data_map, model_map):
        """
        :param param:
        :param data_map:
        :param model_map:
        :return:
        compare train loss with test loss, judge for overfitting or underfitting
        """
        train_feat = "{dp}/bin_feata_toy.dat".format(dp=data_map.get('data_path'))
        eval_feat = "{dp}/bin_featb.dat".format(dp=data_map.get('data_path'))
        model_bin = "{mp}/xgb_model.bin".format(mp=model_map.get('model_path'))
        dtrain = xgb.DMatrix(train_feat)
        deval = xgb.DMatrix(eval_feat)

        num_round = 50
        watchlist = [(dtrain, 'train'), (deval, 'val')]
        bst = xgb.train(param, dtrain, num_round, verbose_eval=1, evals=watchlist)
        # xgb.to_graphviz(bst, num_trees=0)
        # bst.save_model(model_bin)
        # dump_path = model_map.get('dump_path')
        # model2txt(model_bin, dump_path)
        # feature_importance(bst)
        return bst

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
        plt.savefig('model.png', dpi=300)
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

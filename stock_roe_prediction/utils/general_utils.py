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
import tensorflow as tf
import json


def load_data(infile):
    df = pd.read_csv(infile)
    return df


def read_conf(infile):
    fi = codecs.open(infile, 'r', 'utf-8')
    data = fi.read()
    fi.close()
    para_map = yaml.load(data, yaml.FullLoader)
    return para_map


def get_data(in_para):
    train_data_file = "{dp}/train_small.csv".format(dp=in_para.get('data_path'))
    target = 'target'
    df = load_data(train_data_file)
    feats = df.filter(like="f_").columns.tolist()
    return df, feats, target


def get_dataset(in_para, batch_size=32):
    train_data_file = "{dp}/train_small.csv".format(dp=in_para.get('data_path'))
    target_col = 'row_id'
    chunk = pd.read_csv(train_data_file, iterator=True)
    feats = chunk.get_chunk(1).columns.tolist()
    feats_tmp = feats.copy()
    ds = get_dataset_from_csv(train_data_file, target_col, batch_size, feats_tmp)
    print(feats)
    return ds, feats, target_col


def get_dataset_from_csv(csv_file, target_col, batch_size, feat_cols=None):
    if feat_cols is None:
        ds = tf.data.experimental.make_csv_dataset(
            csv_file,
            batch_size=batch_size,
            label_name=target_col,
            na_value="?",
            num_epochs=1,
            ignore_errors=True
        )
    else:
        ds = tf.data.experimental.make_csv_dataset(
          csv_file,
          batch_size=batch_size,
          select_columns=feat_cols,
          label_name=target_col,
          na_value="?",
          num_epochs=1,
          ignore_errors=True
        )
    return ds


def get_sum(df: pd.DataFrame, col):
    return df[col].sum()


def get_mean_of_numerical_var(csv_file_big, out_json_file, chunk_size=10000):
    chunk = pd.read_csv(csv_file_big, iterator=True)
    sample_cnt = 0
    feat_val_sum = {}
    loop = False
    tmp_df = chunk.get_chunk(chunk_size)
    cols = tmp_df.filter(like='f_').columns.tolist()
    for col in cols:
        feat_val_sum.setdefault(col, 0)
    real_size = len(tmp_df)
    tmp_df_ser = get_sum(tmp_df, cols)
    feat_val_sum_tmp = tmp_df_ser.to_dict()
    for key in feat_val_sum_tmp:
        feat_val_sum[key] += feat_val_sum_tmp[key]
    sample_cnt += real_size
    if sample_cnt == chunk_size:
        loop = True
    while loop:
        try:
            tmp_df = chunk.get_chunk(chunk_size)
            real_size = len(tmp_df)
            tmp_df_ser = get_sum(tmp_df, cols)
            feat_val_sum_tmp = tmp_df_ser.to_dict()
            for key in feat_val_sum_tmp:
                feat_val_sum[key] += feat_val_sum_tmp[key]
            sample_cnt += real_size
        except StopIteration:
            loop = False
            print("read finished!")
            break
    print('=================sample_cnt:{0}===================='.format(sample_cnt))
    print(feat_val_sum)
    feat_val_mean = {}
    for key in feat_val_sum:
        val = '%.6f' % (feat_val_sum.get(key) / sample_cnt)
        feat_val_mean.setdefault(key, float(val))
    print(feat_val_mean)
    fo = open(out_json_file, 'w')
    fo.write(json.dumps(feat_val_mean))
    fo.close()
    return feat_val_mean


def get_vals_of_categorical_var(csv_file_big, out_json_file, f_cols=None, chunk_size=10000):
    cate_var2vals = {}
    chunk = pd.read_csv(csv_file_big, iterator=True)
    sample_cnt = 0
    loop = False
    tmp_df = chunk.get_chunk(chunk_size)
    cols = filter(lambda x: x not in f_cols, tmp_df.columns.tolist())
    for col in cols:
        cate_var2vals.setdefault(col, set())
        tmp = set(tmp_df[col][:])
        cate_var2vals[col].update(tmp)
    real_size = len(tmp_df)
    sample_cnt += real_size
    if sample_cnt == chunk_size:
        loop = True
    while loop:
        try:
            tmp_df = chunk.get_chunk(chunk_size)
            real_size = len(tmp_df)
            for col in cols:
                tmp = set(tmp_df[col][:])
                cate_var2vals[col].update(tmp)
            sample_cnt += real_size
        except StopIteration:
            loop = False
            print("read finished!")
            break
    print('=================sample_cnt:{0}===================='.format(sample_cnt))
    print(cate_var2vals)
    cate_var2vals_t = {}
    for key in cate_var2vals:
        cate_var2vals_t.setdefault(key, list(cate_var2vals.get(key)))
    print(cate_var2vals_t)
    fo = open(out_json_file, 'w')
    fo.write(json.dumps(cate_var2vals_t))
    fo.close()
    return cate_var2vals_t


if __name__ == '__main__':
    in_file = '../conf/nn_conf.yaml'
    para_map = read_conf(in_file)
    in_para = para_map.get('in')
    csv_file = '{dp}/train_small.csv'.format(dp=in_para.get('data_path'))
    # feat_json_file = '../data/ubiquant-market-prediction/numerical_feat_small.json'
    # # get_mean_of_numerical_var(csv_file, feat_json_file)
    # csv_file_cate = '../data/titanic/train.csv'
    # cate_feat_json_file = '../data/titanic/categorical_feat.json'
    # cols = ['age', 'fare']
    # get_vals_of_categorical_var(csv_file_cate, cate_feat_json_file, cols, 500)
    reader = pd.read_csv(csv_file, iterator=True)
    tmp_df1 = reader.get_chunk(5)
    tmp_df2 = reader.get_chunk(6)
    print(tmp_df1)
    tmp_df2.reset_index(drop=True, inplace=True)
    print(tmp_df2)

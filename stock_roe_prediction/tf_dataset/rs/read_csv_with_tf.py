"""
==========================
# -*- coding: utf8 -*-
# @Author   : louiss007
# @Time     : 2022/2/2 11:33
# @FileName : read_csv_with_tf.py
# @Email    : quant_master2000@163.com
==========================
"""
import functools

import numpy as np
import tensorflow as tf


train_file_path = '../../data/titanic/train.csv'
test_file_path = '../../data/titanic/eval.csv'

# 让 numpy 数据更易读。
np.set_printoptions(precision=3, suppress=True)

LABEL_COLUMN = 'survived'
# for classification
LABELS = [0, 1]


def get_dataset(csv_file, target_col, batch_size, feat_cols=None):
    if feat_cols is None:
        dataset = tf.data.experimental.make_csv_dataset(
            csv_file,
            batch_size=batch_size,
            label_name=target_col,
            na_value="?",
            num_epochs=1,
            ignore_errors=True
        )
    else:
        dataset = tf.data.experimental.make_csv_dataset(
          csv_file,
          batch_size=batch_size,
          select_columns=feat_cols,
          label_name=target_col,
          na_value="?",
          num_epochs=1,
          ignore_errors=True
        )
    return dataset


raw_train_data = get_dataset(train_file_path, LABEL_COLUMN, 12)
raw_test_data = get_dataset(test_file_path, LABEL_COLUMN, 12)

examples, labels = next(iter(raw_train_data))   # 第一个批次
# print("EXAMPLES: \n", examples, "\n")
# print("LABELS: \n", labels)


def data_preprocessing_for_categorical(categorical_features):
    categorical_columns = []
    for feature, vocab in categorical_features.items():
        cat_col = tf.feature_column.categorical_column_with_vocabulary_list(
            key=feature, vocabulary_list=vocab)
        # categorical_columns.append(tf.feature_column.indicator_column(cat_col))
        categorical_columns.append(cat_col)
    return categorical_columns


def data_preprocessing_for_categorical_v2(categorical_features):
    categorical_columns = data_preprocessing_for_categorical(categorical_features)
    feat_value_cnt = 0
    for feat in categorical_features:
        feat_value_cnt += len(categorical_features.get(feat))

    print('==============feat_value_cnt:{0}=============='.format(feat_value_cnt))
    categorical_columns_embed = []
    for feat_col in categorical_columns:
        cat_col_embed = tf.feature_column.embedding_column(feat_col, dimension=feat_value_cnt)
        categorical_columns_embed.append(cat_col_embed)
        # categorical_columns_embed.append(tf.feature_column.indicator_column(cat_col))
    return categorical_columns_embed


def process_continuous_data(mean, data):
  # 标准化数据
  data = tf.cast(data, tf.float32) * 1/(2*mean)
  return tf.reshape(data, [-1, 1])


def data_preprocessing_for_numerical(numerical_features):
    numerical_columns = []
    for feature in numerical_features:
        num_col = tf.feature_column.numeric_column(
            feature, normalizer_fn=functools.partial(
                process_continuous_data,
                numerical_features[feature]
            )
        )
        numerical_columns.append(num_col)
    return numerical_columns


def build_categorical_feature(values):
    vals = [val.encode() for val in values]
    f = tf.train.Feature(bytes_list=tf.train.BytesList(value=vals))
    return f


def build_numerical_feature(value):
    f = tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
    return f


def build_categorical_features(categorical_features):
    features = {}
    for key in categorical_features:
        values = categorical_features.get(key)
        f = build_categorical_feature(values)
        features.setdefault(key, f)
    return features


def build_numerical_features(numerical_features):
    features = {}
    for key in numerical_features:
        value = numerical_features.get(key)
        f = build_numerical_feature(value)
        features.setdefault(key, f)
    return features


categorical_features = {
    'sex': ['male', 'female'],
    'class': ['First', 'Second', 'Third'],
    'deck': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'],
    'embark_town': ['Cherbourg', 'Southhampton', 'Queenstown'],
    'alone': ['y', 'n']
}


numerical_features = {
    'age': 29.631308,
    'n_siblings_spouses': 0.545455,
    'parch': 0.379585,
    'fare': 34.385399
}

cate_features = build_categorical_features(categorical_features)
nume_features = build_numerical_features(numerical_features)
features = {}
features.update(cate_features)
features.update(nume_features)
tf_example = tf.train.Example(features=tf.train.Features(feature=features))
columns = []
categorical_columns = data_preprocessing_for_categorical_v2(categorical_features)
numerical_columns = data_preprocessing_for_numerical(numerical_features)
columns.extend(categorical_columns)
columns.extend(numerical_columns)
features = tf.io.parse_example([tf_example.SerializeToString()], tf.feature_column.make_parse_example_spec(columns))
inputs = tf.feature_column.input_layer(features=features, feature_columns=columns)
print('=======================shape:{}======================'.format(inputs.get_shape()[1]))

# preprocessing_layer = tf.keras.layers.DenseFeatures(categorical_columns+numerical_columns)

sess = tf.Session()
variables_initner = tf.global_variables_initializer()
tables_initner = tf.tables_initializer()
sess.run(variables_initner)
sess.run(tables_initner)
v = sess.run(inputs)
print(v)
# model = tf.keras.Sequential([
#   preprocessing_layer,
#   tf.keras.layers.Dense(128, activation='relu'),
#   tf.keras.layers.Dense(128, activation='relu'),
#   tf.keras.layers.Dense(1, activation='sigmoid'),
# ])
#
# model.compile(
#     loss='binary_crossentropy',
#     optimizer='adam',
#     metrics=['accuracy']
# )
#
# train_data = raw_train_data.shuffle(500)
# test_data = raw_test_data
# model.fit(train_data, epochs=2)
#
# test_loss, test_accuracy = model.evaluate(test_data)
# print('\n\nTest Loss {}, Test Accuracy {}'.format(test_loss, test_accuracy))
#
# predictions = model.predict(test_data)
# # 显示部分结果
# for prediction, survived in zip(predictions[:10], list(test_data)[0][1][:10]):
#   print("Predicted survival: {:.2%}".format(prediction[0]),
#         " | Actual outcome: ",
#         ("SURVIVED" if bool(survived) else "DIED"))


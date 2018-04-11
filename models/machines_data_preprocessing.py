#!/usr/bin/env python
# coding=utf-8

import os
import csv
import pandas as pd
import numpy as np
import string
import math
import types
import shelve
import pickle
from collections import Counter
from sklearn.feature_extraction import DictVectorizer
import sys
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.utils import np_utils
from sklearn.preprocessing import MultiLabelBinarizer


def open_file(filename, mode='r'):
    return open(filename, mode, encoding='utf-8', errors='ignore')


def process_file(pd_ok, word_to_id, max_length=600):

    pd_data = pd_ok
    selected_features_text = 'content'
    targ = 'polarity'

    X_train_text = pd_data[selected_features_text]
    X_train_text = pd.DataFrame(X_train_text)

    y_train = pd_data[targ]
    y_train = pd.DataFrame(y_train)
    # print(y_train)

    ####################################################################################
    ##y一起放在一个vec
    y_train_max = []
    for idd, row in y_train.iterrows():
        # print(idd, end=",")
        arr = []
        polarity = row[u'polarity']
        polarity = int(str(polarity))
        arr.append(polarity)
        arr = np.array(arr)
        y_train_max.append(arr)
    y_train_max = np.array(y_train_max)

    y_train_labels = y_train_max

    print(y_train_labels[:10])
    print()


    ####################################################################################
    ##text 按maxlen padding
    def handle_text(contents):
        sentences_origin = contents.split()

        ######
        max_id = len(word_to_id)
        tmp_token2id = []

        ###padding 到 maxlen
        if len(sentences_origin) < max_length:
            for ax in range(max_length - len(sentences_origin)):
                sentences_origin.append(" ")
        elif len(sentences_origin) > max_length:
            sentences_origin = sentences_origin[:max_length]

        for token in sentences_origin:
            if token in word_to_id:
                tmp_token2id.append(word_to_id[token])
            else:
                tmp_token2id.append(max_id)

        if len(sentences_origin) != max_length:
            print("len(item_sentences) != max_length")

        tmp_token2id = np.array(tmp_token2id)

        return tmp_token2id

    x_contents = []
    for dx, row in X_train_text.iterrows():
        concat_str = str(row[u'content'])
        contents = handle_text(concat_str)  ##contents 是一个20维的list
        x_contents.append(contents)

    x_contents = np.array(x_contents)

    print(x_contents[:10])
    print()

    return x_contents, y_train_labels


def construct_vocab(pd_ok, vocab_dir, vocab_size):
    pd_data = pd_ok

    X_train_text = pd_data["content"]
    X_train_text = pd.DataFrame(X_train_text)

    all_data = []
    for dx, row in X_train_text.iterrows():
        concat_str = str(row[u'content'])
        strs = concat_str.split()
        all_data.extend(strs)

    counter = Counter(all_data)
    count_pairs = counter.most_common(vocab_size - 1)
    words, _ = list(zip(*count_pairs))
    words = ['<PAD>'] + list(words)

    open_file(vocab_dir, mode='w').write('\n'.join(words) + '\n')


def construct_discrete_vocab(pd_ok, vocab_discrete_dir, vocab_discrete_size):
    pd_data = pd_ok

    selected_features = ["casts", "title", "directors"]
    X_content = pd_data[selected_features]

    all_data = []
    for dx, row in X_content.iterrows():
        casts = str(row[u'casts'])
        casts = casts.split()
        # print(casts)
        all_data.extend(casts)
        # print(str_content)
        directors = str(row[u'directors'])
        directors = directors.split()
        # print(directors)
        all_data.extend(directors)
        title = str(row[u'title'])
        all_data.append(title)
        # print(all_data)

    counter = Counter(all_data)
    count_pairs = counter.most_common(vocab_discrete_size - 1)
    words, _ = list(zip(*count_pairs))
    open_file(vocab_discrete_dir, mode='w').write('\n'.join(words) + '\n')


vocab_size = 113000


def get_data(file_name, save_data_path, maxlen=500, test_len_rate=0.1):
    global vocab_size#, vocab_discrete_size  # , ratio, _positive, _negative

    save_data_path_selected_data = file_name
    pd_ok = read_m_data(save_data_path_selected_data)

    ####################
    print("构建词表中……")
    print()
    vocab_dir = os.path.join(save_data_path, 'vocab_content.txt')
    if not os.path.exists(vocab_dir):
        construct_vocab(pd_ok, vocab_dir, vocab_size)
    # 构造类别表
    # categories = ['0.0', '1.0']
    # cat_to_id = dict(zip(categories, range(len(categories))))

    # 读取vocab_content词汇表
    words = open_file(vocab_dir).read().strip().split('\n')
    word_to_id = dict(zip(words, range(len(words))))
    vocab_size = len(words)

    ####################

    print("构建训练数据中……")
    print()
    x_train_texts, y_train = process_file(pd_ok,word_to_id, maxlen)

    ####################
    print("划分测试集数据中……")
    print()

    len_all = len(x_train_texts)
    (x_test_texts, y_test) = [], []
    (x_train_text_f, y_train_f) = [], []

    len_test = int(len_all * test_len_rate)
    rd_list = np.arange(len_all)
    print("length of training and developing: ", len_all)
    rd_list_test = np.random.choice(rd_list, size=len_test, replace=False)
    # rd_list_train = [i for i in rd_list if i not in rd_list_test]
    for ind in rd_list:
        if ind in rd_list_test:
            x_test_texts.append(x_train_texts[ind])
            y_test.append(y_train[ind])
        else:
            x_train_text_f.append(x_train_texts[ind])
            y_train_f.append(y_train[ind])
    x_test_texts = np.array(x_test_texts)
    y_test = np.array(y_test)
    x_train_text_f = np.array(x_train_text_f)
    y_train_f = np.array(y_train_f)

    ####################
    return_str = "\nx_train_texts_f shape: " + str(x_train_texts.shape)
    # return_str += "\nx_train_feat_f shape: " + str(x_train_feat.shape)
    return_str += "\ny_train_f shape: " + str(y_train.shape)
    return_str += "\n\n"
    print("x_train_texts_f shape: ", x_train_texts.shape)
    # print("x_train_feat_f shape: ", x_train_feat.shape)
    print("y_train_f shape: ", y_train.shape)

    # return words, return_str, (x_train_text_f, x_train_feat_f, y_train_f), (x_test_text, x_test_feat, y_test)
    return return_str, (x_train_text_f, y_train_f),(x_test_texts, y_test)


def read_m_data(read_path, encoding='utf-8'):
    df_all_raw = pd.read_csv(read_path, encoding=encoding, sep="\t", dtype=str)
    return df_all_raw


def save_data(data_df, result_file_dir):
    data_df = data_df.reset_index(drop=True)
    data_df.to_csv(result_file_dir, encoding="utf-8", index=False)

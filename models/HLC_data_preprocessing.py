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


def process_file(pd_ok, word_to_id, word_discrete_to_id, max_length=600, gb_len=50):
    pd_data = pd_ok
    selected_features_text = 'content'
    selected_features_num = ["casts", "title", "rate", "directors", "run_length"]
    targ = 'polarity'

    X_train_text = pd_data[selected_features_text]
    X_train_text = pd.DataFrame(X_train_text)

    X_train_num = pd_data[selected_features_num]
    y_train = pd_data[targ]
    y_train = pd.DataFrame(y_train)
    
    ####################################################################################
    ##numeracal 一起放在一个矩阵
    data_num_id = []
    contents_num = []
    for dx, row in X_train_num.iterrows():
        str_content = []
        casts = str(row[u'casts'])
        casts = casts.split()
        # print(casts)
        str_content.extend(casts)
        # print(str_content)
        directors = str(row[u'directors'])
        directors = directors.split()
        # print(directors)
        str_content.extend(directors)
        title = str(row[u'title'])
        str_content.append(title)
        # print(str_content)
        contents_num.append(str_content)
    for i in range(len(contents_num)):
        data_num_id.append([word_discrete_to_id[x] for x in contents_num[i] if x in word_discrete_to_id])
        # print(data_num_id[i])

    ##numeracal 一起放在一个矩阵
    X_train_num_matrix = []
    for idd, row in X_train_num.iterrows():
        arr = []
        rate = row[u'rate']
        rate = int(float(str(rate)) * 10)
        arr.append(rate)
        run_length = row[u'run_length']
        run_length = int(str(run_length))
        arr.append(run_length)

        # print(data_num_id[idd])
        arr.extend(data_num_id[idd])
        # print(arr)

        # arr = np.array(arr)
        X_train_num_matrix.append(arr)
    X_train_num = sequence.pad_sequences(X_train_num_matrix, gb_len)
    print(X_train_num[:10])
    print()
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

    ### 二维，可以提升预测准确度！
    y_train_labels = np_utils.to_categorical(y_train_max)

    print(y_train_labels[:10])
    print()

    ####################################################################################
    def partition(contents):
        contents = contents.replace("。", "\n").replace("？", "\n").replace("：", "\n") \
            .replace("…", "\n").replace("!", "\n").replace(".", "\n") \
            .replace(";", "\n").replace(":", "\n").replace("！", "\n") \
            .replace("；", "\n").replace("?", "\n")
        
        sentences_origin = contents.split("\n")
        sentences_unsorted = []
        for item in sentences_origin:
            if len(item) == 0:
                continue
            sentence = item.split()
            sentences_unsorted.append(sentence)
        sentences_sorted = sorted(sentences_unsorted, key=lambda x: len(x), reverse=True)

        sentences_sorted2 = []
        for item in sentences_sorted:
            if len(item) != 0:
                sentences_sorted2.append(item)
        sentences_sorted = sentences_sorted2

        sentence_length = len(sentences_sorted)

        potitioned_sentences = []
        if sentence_length < 20:
            ##
            if sentence_length == 1 or sentence_length == 2 or sentence_length == 10 or sentence_length == 4 or sentence_length == 5:
                for item_s in range(sentence_length):
                    tmp = div_list(sentences_sorted[item_s], 20 // sentence_length)
                    potitioned_sentences.extend(tmp)
            elif sentence_length == 3:
                for item_s in range(2):
                    tmp = div_list(sentences_sorted[item_s], 10)
                    potitioned_sentences.extend(tmp)
                # try:
                potitioned_sentences[len(potitioned_sentences) - 1].extend(
                    sentences_sorted[len(sentences_sorted) - 1])
                # except:
                #     print(sentences_sorted)
                #     print(potitioned_sentences)
            elif sentence_length == 6 or sentence_length == 7 or sentence_length == 8 or sentence_length == 9:
                for item_s in range(5):
                    tmp = div_list(sentences_sorted[item_s], 4)
                    potitioned_sentences.extend(tmp)
                for item_cmpst in range(sentence_length - 5):
                    potitioned_sentences[len(potitioned_sentences) - (1 + item_cmpst)].extend(
                        sentences_sorted[len(sentences_sorted) - (1 + item_cmpst)])
            else:  # 超过10的都补齐前面
                for item_s in range(10):
                    tmp = div_list(sentences_sorted[item_s], 2)
                    potitioned_sentences.extend(tmp)
                for item_cmpst in range(sentence_length - 10):
                    potitioned_sentences[len(potitioned_sentences) - (1 + item_cmpst)].extend(
                        sentences_sorted[len(sentences_sorted) - (1 + item_cmpst)])

            if len(potitioned_sentences) < 20:  # 不足20句，加空白串10串构成一句
                pads = [" " for iii in range(max_length)]
                leng = len(potitioned_sentences)
                for k in range(20 - leng):
                    potitioned_sentences.append(pads)

        elif sentence_length > 20:
            ##
            potitioned_sentences = div_list_extend(sentences_sorted, 20)
        else:
            ##
            potitioned_sentences = sentences_sorted

        ######
        potitioned_sentences3 = []
        max_id = len(word_to_id)
        for item_sentences in potitioned_sentences:
            tmp_token2id = []

            ###padding 到 maxlen
            if len(item_sentences) < max_length:
                for ax in range(max_length - len(item_sentences)):
                    item_sentences.append(" ")
            elif len(item_sentences) > max_length:
                item_sentences = item_sentences[:max_length]

            for token in item_sentences:
                if token in word_to_id:
                    tmp_token2id.append(word_to_id[token])
                else:
                    tmp_token2id.append(max_id)
            if len(item_sentences) != max_length:
                print("len(item_sentences) != max_length")
            tmp_token2id = np.array(tmp_token2id)
            potitioned_sentences3.append(tmp_token2id)
        # potitioned_sentences3 = np.array(potitioned_sentences3)

        if len(potitioned_sentences3) != 20:
            print("len(potitioned_sentences3) != 20")

        return sentence_length, potitioned_sentences3

    def div_list_extend(ls, n):
        def ferry(main, sub, ind):
            tmp_sub = []
            tmp_main = main
            if ind % 2 == 0:
                # tmp_sub = sub.reverse()
                tmp_sub = list(reversed(sub))
            else:
                tmp_sub = sub
            for item in range(len(main)):
                if (item + 1) > len(tmp_sub):
                    break
                tmp_main[item].extend(tmp_sub[item])
            return tmp_main

        ls_len = len(ls)
        if n <= 0 or 0 == ls_len:
            return []

        if n > ls_len:
            return []
        elif n == ls_len:
            return [[i] for i in ls]

        else:
            j = ls_len // n
            k = ls_len % n

            tmp_all = ls[: 20]
            ind = 1
            for jj in range(20, (n - 1) * j, j):
                tmp_all = ferry(tmp_all, ls[jj: jj + j], ind)
                ind += 1
            tmp_all = ferry(tmp_all, ls[(n - 1) * j:], ind)

            return tmp_all

    def div_list(ls, n):
        if not isinstance(ls, list) or not isinstance(n, int):
            return []
        ls_len = len(ls)
        if n <= 0 or 0 == ls_len:
            return []
        if n >= ls_len:
            return [[i] for i in ls]
        else:
            j = ls_len // n
            k = ls_len % n
            ### j,j,j,...(前面有n-1个j),j+k
            # 步长j,次数n-1
            ls_return = []
            for i in range(0, (n - 1) * j, j):
                ls_return.append(ls[i:i + j])
                # 算上末尾的j+k
            ls_return.append(ls[(n - 1) * j:])
            return ls_return

    x_contents1 = list()
    x_contents2 = list()
    x_contents3 = list()
    x_contents4 = list()
    x_contents5 = list()
    x_contents6 = list()
    x_contents7 = list()
    x_contents8 = list()
    x_contents9 = list()
    x_contents10 = list()
    x_contents11 = list()
    x_contents12 = list()
    x_contents13 = list()
    x_contents14 = list()
    x_contents15 = list()
    x_contents16 = list()
    x_contents17 = list()
    x_contents18 = list()
    x_contents19 = list()
    x_contents20 = list()
    for dx, row in X_train_text.iterrows():
        concat_str = str(row[u'content'])
        sentence_length, contents = partition(concat_str)  ##contents 是一个20维的list
        if len(contents) != 20:
            print(str(dx), " length of potitioned_sentences:", len(contents))
            print(sentence_length)
            print()
        # D = np.vstack((D, contents))
        # for ix in range(20):
        x_contents1.append(contents[0])
        x_contents2.append(contents[1])
        x_contents3.append(contents[2])
        x_contents4.append(contents[3])
        x_contents5.append(contents[4])
        x_contents6.append(contents[5])
        x_contents7.append(contents[6])
        x_contents8.append(contents[7])
        x_contents9.append(contents[8])
        x_contents10.append(contents[9])
        x_contents11.append(contents[10])
        x_contents12.append(contents[11])
        x_contents13.append(contents[12])
        x_contents14.append(contents[13])
        x_contents15.append(contents[14])
        x_contents16.append(contents[15])
        x_contents17.append(contents[16])
        x_contents18.append(contents[17])
        x_contents19.append(contents[18])
        x_contents20.append(contents[19])
    # x_contents = np.vstack(x_contents)


    x_contents1 = np.array(x_contents1)
    x_contents2 = np.array(x_contents2)
    x_contents3 = np.array(x_contents3)
    x_contents4 = np.array(x_contents4)
    x_contents5 = np.array(x_contents5)
    x_contents6 = np.array(x_contents6)
    x_contents7 = np.array(x_contents7)
    x_contents8 = np.array(x_contents8)
    x_contents9 = np.array(x_contents9)
    x_contents10 = np.array(x_contents10)
    x_contents11 = np.array(x_contents11)
    x_contents12 = np.array(x_contents12)
    x_contents13 = np.array(x_contents13)
    x_contents14 = np.array(x_contents14)
    x_contents15 = np.array(x_contents15)
    x_contents16 = np.array(x_contents16)
    x_contents17 = np.array(x_contents17)
    x_contents18 = np.array(x_contents18)
    x_contents19 = np.array(x_contents19)
    x_contents20 = np.array(x_contents20)

    x_contents_texts = [x_contents1, x_contents2, x_contents3, x_contents4, x_contents5,
                        x_contents6, x_contents7, x_contents8, x_contents9, x_contents10,
                        x_contents11, x_contents12, x_contents13, x_contents14, x_contents15,
                        x_contents16, x_contents17, x_contents18, x_contents19, x_contents20
                        ]

    # 现在的shape是：(12280, 20, 512)
    ##reshape成分别的20份 batch组成20份：20 x （12280, 512）,从aixs=1拆分为20份，装到list中

    print(x_contents_texts[:10])
    print()
    return x_contents_texts, X_train_num, y_train_labels
    

def construct_vocab(embedding_file, vocab_dir, vocab_size):
    """根据训练集构建词汇表，存储"""
    words = []
    f = open(embedding_file)
    ii = 0
    for line in f:
        # print(line)
        if ii == 0:
            ii += 1
            continue
        values = line.split()
        # print(values)
        word = values[0]
        words.append(word)
        ii += 1
    f.close()
    # print(words)

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
vocab_discrete_size = 10000


def get_data(file_name, embedding_file, save_data_path, test_len=32, maxlen=500, gb_len=50):
    global vocab_size, vocab_discrete_size  # , ratio, _positive, _negative

    save_data_path_selected_data = file_name
    pd_ok = read_m_data(save_data_path_selected_data)

    ####################
    print("构建词表中……")
    print()
    vocab_dir = os.path.join(save_data_path, 'vocab_content.txt')
    if not os.path.exists(vocab_dir):
        construct_vocab(embedding_file, vocab_dir, vocab_size)
    vocab_discrete_dir = os.path.join(save_data_path, 'vocab_discrete.txt')
    if not os.path.exists(vocab_discrete_dir):
        construct_discrete_vocab(pd_ok, vocab_discrete_dir, vocab_discrete_size)

    # 构造类别表
    # categories = ['0.0', '1.0']
    # cat_to_id = dict(zip(categories, range(len(categories))))

    # 读取vocab_content词汇表
    words = open_file(vocab_dir).read().strip().split('\n')
    word_to_id = dict(zip(words, range(len(words))))
    vocab_size = len(words)

    # vocab_discrete
    words_discrete = open_file(vocab_discrete_dir).read().strip().split('\n')
    word_discrete_to_id = dict(zip(words_discrete, range(len(words_discrete))))
    vocab_discrete_size = len(word_discrete_to_id)
    ####################

    print("构建训练数据中……")
    print()
    ####把处理好的对象持久化，下次直接存取
    train_texts0_file_name = 'train_texts0'
    train_feats_file_name = 'train_feats.pk'
    train_y_file_name = 'train_y.pk'
    train_texts0_file_path = os.path.join(save_data_path, train_texts0_file_name)
    train_feats_file_path = os.path.join(save_data_path, train_feats_file_name)
    train_y_file_path = os.path.join(save_data_path, train_y_file_name)

    x_train_texts = []
    if not os.path.exists(train_texts0_file_path):
        print("从0构建数据……")
        print()
        x_train_texts, x_train_feat, y_train = process_file(pd_ok, word_to_id, word_discrete_to_id, maxlen, gb_len)

        # '''
        train_feats_file = open(train_feats_file_path, 'wb')
        pickle.dump(x_train_feat, train_feats_file)
        train_feats_file.close()
        train_y_file = open(train_y_file_path, 'wb')
        pickle.dump(y_train, train_y_file)
        train_y_file.close()
        # '''

        # shelve.dump(x_train_texts[0], train_texts_file)
        for it in range(20):
            train_texts_file_name = 'train_texts' + str(it)
            train_texts_file_path = os.path.join(save_data_path, train_texts_file_name)
            train_texts_file = open(train_texts_file_path, 'wb')
            pickle.dump(x_train_texts[it], train_texts_file)
            # train_texts_file["x_train_texts[" + str(it) + "]"] = x_train_texts[it]
            train_texts_file.close()
    else:
        print("从本地加载数据……")
        print()

        # train_texts_file = shelve.open(train_texts_file_path)
        for it in range(20):
            train_texts_file_name = 'train_texts' + str(it)
            train_texts_file_path = os.path.join(save_data_path, train_texts_file_name)
            train_texts_file = open(train_texts_file_path, 'rb')
            x_train_text_ = pickle.load(train_texts_file)
            # x_train_texts[it] = train_texts_file["x_train_texts[" + str(it) + "]"]
            x_train_texts.append(x_train_text_)
            train_texts_file.close()

        # '''
        train_feats_file = open(train_feats_file_path, 'rb')
        x_train_feat = pickle.load(train_feats_file)
        train_feats_file.close()
        train_y_file = open(train_y_file_path, 'rb')
        y_train = pickle.load(train_y_file)
        train_y_file.close()
        # '''

    ####################
    # 划分出一部分为测试集
    print("没有测试集数据，只有训练集……")
    # print("划分测试集数据中……")
    print()
    
    return_str = "\nx_train_texts_f shape[0]: " + str(x_train_texts[0].shape)
    return_str += "\nx_train_feat_f shape: " + str(x_train_feat.shape)
    return_str += "\ny_train_f shape: " + str(y_train.shape)
    return_str += "\n\n"
    print("x_train_texts_f[0] shape: ", x_train_texts[0].shape)
    print("x_train_feat_f shape: ", x_train_feat.shape)
    print("y_train_f shape: ", y_train.shape)

    return words, return_str, (x_train_texts, x_train_feat, y_train)


def read_m_data(read_path, encoding='utf-8'):
    df_all_raw = pd.read_csv(read_path, encoding=encoding, sep="\t", dtype=str)
    return df_all_raw


def save_data(data_df, result_file_dir):
    data_df = data_df.reset_index(drop=True)
    data_df.to_csv(result_file_dir, encoding="utf-8", index=False)

# !/usr/bin/env python
# coding=utf-8

from gensim.models import *
import logging

import os
import csv
import pandas as pd
import numpy as np
import string
import math
import types
import re
from collections import Counter
from sklearn.feature_extraction import DictVectorizer
import sys
from keras.preprocessing import sequence
from keras.utils import to_categorical


def segment_sentence(review_pd, corpus_file):
    k = 0
    review_corpus = ""
    for key, value1 in review_pd.iterrows():
        # if (key % 20 == 0):
        print(k, end=",")
        k += 1
        # print(value1)
        content = value1[u'content']
        # if content is None:
        #     print("null")
            # print(content)
        content = content.replace("。","\n").replace("？","\n").replace("：","\n")\
            .replace("…", "\n").replace("!", "\n").replace(".", "\n")\
            .replace(";", "\n").replace(":", "\n").replace("！", "\n")\
            .replace("；", "\n").replace("?", "\n")
        review_corpus += (content + "\n")
        # m_new_table_item = {u'content': content
        #                     }
        # review_corpus[str(key)] = m_new_table_item

    fout = open(corpus_file, 'w')
    fout.write(review_corpus)
    fout.close()

def training(corpus_file,w2v_model_file,w2v_file):

    logging.basicConfig(filename='w2v.log', filemode="w",format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    sentences = word2vec.Text8Corpus(corpus_file)  # 加载语料
    model = word2vec.Word2Vec(sentences, size=200, sg=1, window=5, hs=1, workers=4)  # 默认window=5

    # 保存模型，以便重用
    model.save(w2v_model_file)
    model.wv.save_word2vec_format(w2v_file, binary=False)
    

def read_m_data(read_path, encoding='utf-8'):
    df_all_raw = pd.read_csv(read_path, encoding=encoding, sep="\t", dtype=str)
    return df_all_raw


def save_data(data_df, result_file_dir):
    data_df = data_df.reset_index(drop=True)
    data_df.to_csv(result_file_dir, encoding="utf-8", sep="\t", index=False)


if __name__ == "__main__":

    reviews_file = r'../raw/handled_reviews.csv'
    corpus_file = r'corpus.txt'
    w2v_file = r'embeddings-200d.txt'
    w2v_model_file = r'embeddings-200d.txt'

    review_pd = pd.read_csv(open(reviews_file, 'rU'), sep="\t", dtype=str)
    # print(review_pd[:3])

    log_file = open("message.log", "w")
    sys.stdout = log_file

    segment_sentence(review_pd, corpus_file)
    pd_handled = training(corpus_file, w2v_model_file,w2v_file)
    save_data(pd_handled, reuslt_review)

    log_file.close()


#!/usr/bin/env python
# coding=utf-8

import os
import csv
import pandas as pd
import numpy as np
import string
import math
import types
import re
import jieba
from collections import Counter
from sklearn.feature_extraction import DictVectorizer
import sys
from keras.preprocessing import sequence
from keras.utils import to_categorical


###做以下工作：
# 1. 数据清洗：去掉html符号,用空格代替\t,emoji
#     <a></a>
#      <p>
#       </p>
#     <div></div>
# 2. 记录加上对应的电影背景信息，
# 3. 为每条文本分词

def handle(review_pd, movie_pd):
    def cleaning(text):
        # try:
        text = text.replace("<p>", "").replace("</p>", "")
        emoji_pattern = re.compile(u'[\U00010000-\U0010ffff]')
        text = emoji_pattern.sub(r'', text)

        res = r'<a .*?>(.*?)</a>'
        text = re.sub(res, "", text)

        res2 = r'<div .*?>(.*?)</div>'
        text = re.sub(res2, "", text)
        text = text.replace("</div>", "")

        text = text.replace("&amp;", "").replace("—", "").replace("&nbsp;", "") \
            .replace("&quot;", "").replace("~", "").replace("--", "").replace("&lt;", "") \
            .replace("&gt;", "").replace("=", "").replace("\n", "").replace("\r", "").replace("\n\r", "")

        return text

    def segments(text):
        # 把停用词做成字典
        stopwords = {}
        fstop = open('stopwords.txt', 'r')
        for eachWord in fstop:
            stopwords[eachWord.strip()] = eachWord.strip()
        fstop.close()

        # jieba.enable_parallel(4)  # 并行分词
        # line = text.strip()  # 去除每行首尾可能出现的空格，并转为Unicode进行处理
        line1 = re.sub(r'[0-9+\.\/_,$%^*-+\"\']+|[+——！，、~@#￥%&*（）]+',"", text)
        wordList = list(jieba.cut(line1))  # 用结巴分词，对每行内容进行分词
        outStr = ''
        # print wordList
        for word in wordList:
            if word not in stopwords:
                outStr += word
                outStr += ' '
        # outStr = outStr.replace("=", "")
        outStr = ' '.join(outStr.split())
        return outStr

    def replace_black(casts):
        return casts.replace(";", " ")

    print("\n\n\n")
    print("转录到dict")
    print("\n\n\n")
    k = 0
    new_main_review = {}
    for key, value1 in review_pd.iterrows():
        if (key % 20 == 0):
            print(k, end=",")
        k += 1
        # print(value1)
        polarity = value1[u'polarity']
        movie_id = value1[u'movie_id']
        content = value1[u'content']

        m_new_table_item = {u'polarity': polarity, u'movie_id': movie_id,
                            u'content': content
                            }
        new_main_review[str(key)] = m_new_table_item

    k = 0
    new_main_movie = {}
    for key, value1 in movie_pd.iterrows():
        if (key % 20 == 0):
            print(k, end=",")
        k += 1
        # print(value1)
        casts = value1[u'casts']
        movie_id = value1[u'movie_id']
        title = value1[u'title']
        rate = value1[u'rate']
        directors = value1[u'directors']
        run_length = value1[u'run_length']

        m_new_table_item = {u'casts': casts,
                            u'title': title, u'rate': rate,
                            u'directors': directors, u'run_length': run_length
                            }
        new_main_movie[str(movie_id)] = m_new_table_item

    print("\n\n\n")
    print("正式清理工作")
    print("length: "+str(len(new_main_review)))
    print("\n\n\n")
    new_review_table = dict()
    isi = 0
    for dx1, value1 in new_main_review.items():
        # if (isi % 20) == 0:
        print(isi, end=' ')
        isi += 1

        polarity = value1[u'polarity']
        movie_id = value1[u'movie_id']
        content = value1[u'content']

        try:
            content = cleaning(content)
            content = segments(content)
        except:
            continue

        movie_dict = new_main_movie[str(movie_id)]
        casts = movie_dict[u'casts']
        casts = replace_black(casts)
        directors = movie_dict[u'directors']
        directors = replace_black(directors)
        # movie_id = value1[u'movie_id']
        title = movie_dict[u'title']
        rate = movie_dict[u'rate']
        rate = filter(lambda ch: ch in '0123456789.', rate)
        rate = str("".join(list(rate)))
        run_length = movie_dict[u'run_length']
        run_length = filter(str.isdigit, run_length)
        run_length = str("".join(list(run_length)))
        new_record = {"polarity": polarity, "content": content,
                      u'casts': casts,
                      u'title': title, u'rate': rate,
                      u'directors': directors, u'run_length': run_length
                      }
        new_review_table[str(isi)] = new_record
        # print(new_record)
        # if isi == 50:
        #     return None

    new_review_table_df = dict2pd(new_review_table)
    # all_result_df = old_table.append(new_table_df, ignore_index=True)
    return new_review_table_df


def dict2pd(table):
    df_result = pd.DataFrame(columns={"polarity", "content",
                                      u'casts', u'title', u'rate',
                                      u'directors', u'run_length'
                                      })
    polarity = []
    content = []
    casts = []
    title = []
    rate = []
    directors = []
    run_length = []
    j = 0
    for key, item in table.items():
        if (j % 20 == 0):
            print(j, end=",")
        j += 1

        polarity.append(item["polarity"])
        content.append(item["content"])
        casts.append(item["casts"])
        title.append(item["title"])
        rate.append(item["rate"])
        directors.append(item["directors"])
        run_length.append(item["run_length"])

    data_content = {"polarity": polarity, "content": content,
                    u'casts': casts,
                    u'title': title, u'rate': rate,
                    u'directors': directors, u'run_length': run_length
                    }
    df_result = pd.DataFrame(data_content)
    df_result = df_result.reset_index(drop=True)
    return df_result


def read_m_data(read_path, encoding='utf-8'):
    df_all_raw = pd.read_csv(read_path, encoding=encoding, sep="\t", dtype=str)
    return df_all_raw


def save_data(data_df, result_file_dir):
    data_df = data_df.reset_index(drop=True)
    data_df.to_csv(result_file_dir, encoding="utf-8", sep="\t", index=False)


if __name__ == "__main__":
    movie_pd_file = r'movies.csv'
    review_pd_file = r'reviews.csv'
    reuslt_review = r'handled_reviews.csv'

    movie_pd = read_m_data(movie_pd_file)
    print(movie_pd[:3])
    print()
    # review_pd = read_m_data(review_pd_file)
    review_pd = pd.read_csv(open(review_pd_file, 'rU'), sep="\t", dtype=str)
    print(review_pd[:3])

    pd_handled = handle(review_pd, movie_pd)
    save_data(pd_handled, reuslt_review)

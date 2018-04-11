#!/usr/bin/env python
# coding=utf-8

import os
import csv
import pandas as pd
import numpy as np
import string

def count_max(review_pd, seg_file):
    segwords = {}
    fseg = open(seg_file, 'r')
    for eachWord in fseg:
        segwords[eachWord.strip()] = eachWord.strip()
    fseg.close()

    k = 0
    max_len = 0
    lengs = []
    for key, value1 in review_pd.iterrows():
        # if (key % 20 == 0):
        print(k, end=",")
        k += 1
        content = value1[u'content']
        content = content.replace("。","\n").replace("？","\n").replace("：","\n")\
            .replace("…", "\n").replace("!", "\n").replace(".", "\n")\
            .replace(";", "\n").replace(":", "\n").replace("！", "\n")\
            .replace("；", "\n").replace("?", "\n")
        leng = content.count("\n")
        lengs.append(leng)
        if leng > max_len:
            max_len = leng
    print(max_len+1)
    print()
    print(lengs)

def read_m_data(read_path, encoding='utf-8'):
    df_all_raw = pd.read_csv(read_path, encoding=encoding, sep="\t", dtype=str)
    return df_all_raw


def save_data(data_df, result_file_dir):
    data_df = data_df.reset_index(drop=True)
    data_df.to_csv(result_file_dir, encoding="utf-8", sep="\t", index=False)



if __name__ == "__main__":

    reviews_file = r'./raw/handled_reviews.csv'
    seg_file = r'./embedded/sengment_token.txt'
    review_pd = pd.read_csv(open(reviews_file, 'rU'), sep="\t", dtype=str)

    count_max(review_pd,seg_file)


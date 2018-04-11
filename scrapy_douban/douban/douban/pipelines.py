# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://doc.scrapy.org/en/latest/topics/item-pipeline.html
from douban.items import *
import codecs
import pandas as pd
import os

class DoubanPipeline(object):
    def __init__(self):
        # self.csv_path = '/Users/scofield/workplaces/pythons/deep_networks/scott_trials/nn_for_paper/rte/data/hotel_review_booking/hotel_review_booking/hotel_review_booking/bj_items.csv'
        # self.csv_path = r'/Users/scofield/workplaces/pythons/deep_networks/scott_trials/nn_for_paper/rte/data/hotel_review_booking/hotel_review_booking/hotel_review_booking/sh_items.csv'

        self.base_data_path = r'/Users/scofield/workplaces/pythons/toys/crawler/scapy_trial/douban_graguate_paper/douban/douban/'
        self.file1_name = r'movies1.csv'
        self.file2_name = r'reviews1.csv'
        self.pd1_file = pd.DataFrame(
            columns={"title", "rate", "run_length", 'casts', 'directors', 'movie_id'})
        self.pd2_file = pd.DataFrame(
            columns={"content", "veto", "vote", 'stars', 'polarity', 'movie_id'})

    def process_item(self, item, spider):
        # print(item)
        if isinstance(item, DoubanMovieItem):
            title = item['title']
            rate = item['rate']
            casts = item['casts']
            directors = item['directors']
            length = item['length']
            length = length.replace("分钟","").replace(" ","")
            movie_id = item['movie_id']

            new_record = {"title": title,
                          "rate": rate,
                          "casts": casts,
                          "run_length": length,
                          "directors": directors,
                          "movie_id": movie_id,
                          }
            self.pd1_file = self.pd1_file.append(new_record, ignore_index=True)
            return item
        if isinstance(item, DoubanReviewItem):
            content = item['content']
            content = content.replace("<br>","").replace("&quot;","") \
                .replace("&nbsp;", "").replace("\n","")
            veto = item['veto']
            vote = item['vote']
            stars = item['stars']
            polarity = item['polarity']
            id = item['movie_id']

            new_record = {"content": content,
                          "veto": veto,
                          "vote": vote,
                          "stars": stars,
                          "polarity": polarity,
                          "movie_id": id,
                          }
            self.pd2_file = self.pd2_file.append(new_record, ignore_index=True)
            return item

    def close_spider(self, spider):
        # print(self.pd_file[:10])
        print(len(self.pd1_file))
        self.pd1_file = self.pd1_file.reset_index(drop=True)
        # print(pd_clusterd_result)
        # if not os.path.exists(self.base_data_path+self.file_name):
        #     os.mkdir(self.base_data_path+self.file_name)
        self.pd1_file.to_csv(self.base_data_path + self.file1_name, sep="\t", index=False, header=True, encoding="utf-8")

        print(len(self.pd2_file))
        self.pd2_file = self.pd2_file.reset_index(drop=True)
        # print(pd_clusterd_result)
        # if not os.path.exists(self.base_data_path+self.file_name):
        #     os.mkdir(self.base_data_path+self.file_name)
        self.pd2_file.to_csv(self.base_data_path + self.file2_name, sep="\t", index=False, header=True, encoding="utf-8")

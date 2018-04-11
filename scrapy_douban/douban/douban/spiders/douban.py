import re
import scrapy
from bs4 import BeautifulSoup
from scrapy.selector import Selector
from scrapy.http import Request
from douban.items import DoubanMovieItem, DoubanReviewItem
import requests
import json
import time


class MyCrawler(scrapy.Spider):
    name = 'douban'
    allowd_domain = 'movie.douban.com'
    pure_base_url = 'https://movie.douban.com/tag/#/'

    movie_list_index = 'https://movie.douban.com/j/new_search_subjects?sort=T&range=0,10&tags=&start='

    review_list_1 = 'https://movie.douban.com/subject/'
    review_list_3 = '/reviews?rating='
    # review_5 = '/full'

    review_str_1 = 'https://movie.douban.com/j/review/'
    review_str_3 = '/full'

    movie_count = 600

    # pattern = r'(\d{4})年(\d{1,2})月(\d{1,2})日'
    # pattern_compiled = re.compile(pattern)


    def start_requests(self):

        for i in range(0, self.movie_count, 20):
            url_list = self.movie_list_index + str(i)
            yield Request(url_list, callback=self.start_request_json)

    def start_request_json(self, response):
        file = json.loads(response.body)
        # file = requests.get(url_list).json()  # 这里跟之前的不一样，因为返回的是 json 文件
        # time.sleep(1)
        for ij in range(20):
            dict = file['data'][ij]  # 取出字典中 'data' 下第 [i] 部电影的信息
            urlname = dict['url']
            title = dict['title']
            rate = dict['rate']
            id = dict['id']

            casts = dict['casts']
            casts_strs = ""
            for item in casts:
                casts_strs += (str(item) + ";")
            casts_strs = casts_strs[:-1]

            directors = dict['directors']
            directors_strs = ""
            for item in directors:
                directors_strs += (str(item) + ";")
            directors_strs = directors_strs[:-1]
            print(urlname)
            yield Request(urlname, callback=self.get_movie_page_main, meta={'title': title,
                                                                            'rate': rate,
                                                                            'id': id,
                                                                            'casts': casts_strs,
                                                                            'directors': directors_strs,
                                                                            })

    def get_movie_page_main(self, response):
        main_page_infos = response.xpath("//div[@id='info']").extract()
        content_infos_sel = Selector(text=main_page_infos[0])

        length_infos = content_infos_sel.xpath("//span[@property='v:runtime']/text()").extract()
        length_infos = length_infos[0]
        # length_infos = content_infos_sel.xpath("//span[@property='v:runtime']").extract()

        title = response.meta['title']
        rate = response.meta['rate']
        casts = response.meta['casts']
        directors = response.meta['directors']
        id = response.meta['id']

        item = DoubanMovieItem()
        item['title'] = title
        item['rate'] = rate
        item['casts'] = casts
        item['directors'] = directors
        item['length'] = length_infos
        item['movie_id'] = id
        yield item
        print("movie id: ", id)

        for rat in range(1, 6):
            url_review_list = ""
            url_review_list = self.review_list_1 + id + self.review_list_3 + str(rat)
            yield Request(url_review_list, callback=self.get_one_review, meta={'rate': rat, "movie_id": id})

    def get_one_review(self, response):
        rat = response.meta['rate']
        id = response.meta['movie_id']

        review_list = response.xpath("//div[@class='review-list  ']").extract()
        review_list = review_list[0]
        review_ite1 = Selector(text=review_list)
        review_id2 = review_ite1.xpath("//div[@typeof='v:Review']").extract()
        for review_item in review_id2:
            review_ite = Selector(text=review_item)
            review_id = review_ite.xpath("//@id").extract()
            review_id = review_id[0]
            # print("review_id: ",review_id)
            # review_ite2 = Selector(text=review_type)
            # review_id = review_ite.xpath("//div[@typeof='v:Review']").extract()
            # print("review_id: ", review_id)
            url_review = ""
            url_review = self.review_str_1 + str(review_id) + self.review_str_3
            print(url_review)
            # dict = requests.get(url_review).json()

            yield Request(url_review, callback=self.get_one_review_json,
                          meta={'rate': rat, 'movie_id': id, 'review_id': review_id})

    def get_one_review_json(self, response):
        rat = response.meta['rate']
        id = response.meta['movie_id']
        review_id = response.meta['review_id']
        # time.sleep(2)
        # for ij in range(20):
        # dict = file['data']
        dict = json.loads(response.body)
        content = dict['html']
        stars = rat
        votes = dict['votes']
        # for k in range(6):
        vote = votes['useful_count']
        veto = votes['useless_count']

        if stars > 3:
            polarity = 1
        elif stars == 3:
            polarity = 0
        else:
            polarity = -1

        item = DoubanReviewItem()
        item['content'] = content
        item['veto'] = veto
        item['vote'] = vote
        item['stars'] = stars
        item['polarity'] = polarity
        item['movie_id'] = id
        print("review id: ", review_id)
        yield item


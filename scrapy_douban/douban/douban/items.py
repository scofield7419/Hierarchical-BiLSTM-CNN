# -*- coding: utf-8 -*-

import scrapy


class DoubanMovieItem(scrapy.Item):
    title = scrapy.Field()
    directors = scrapy.Field()
    movie_id = scrapy.Field()
    rate = scrapy.Field()
    # show_date = scrapy.Field()
    # type = scrapy.Field()
    casts = scrapy.Field()
    length = scrapy.Field()
    # stars = scrapy.Field()


class DoubanReviewItem(scrapy.Item):
    content = scrapy.Field()
    movie_id = scrapy.Field()
    veto = scrapy.Field()
    vote = scrapy.Field()
    stars = scrapy.Field()
    polarity = scrapy.Field()

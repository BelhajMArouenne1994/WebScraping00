# -*- coding: utf-8 -*-
import scrapy
from scrapy.loader import ItemLoader
from scrapy.http import FormRequest
from automobile_propre.items import Review

import re
def cleanresponse(raw_html):
    cleanr = re.compile('<div class="quotetitle">.*?</div>')
    cleantext = re.sub(cleanr, ' ', raw_html)
    return cleantext


def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', raw_html)
    return cleantext


# replace multiple occurrences of a character by a single character
def replace(s, ch):
    new_str = []
    l = len( s )

    for i in range( len( s ) ):
        if (s[i] == ch and i != (l - 1) and
                i != 0 and s[i + 1] != ch and s[i - 1] != ch):
            new_str.append( s[i] )

        elif s[i] == ch:
            if ((i != (l - 1) and s[i + 1] == ch) and
                    (i != 0 and s[i - 1] != ch)):
                new_str.append( s[i] )
        else:
            new_str.append( s[i] )
    return ("".join( i for i in new_str ))


class CarsSpider(scrapy.Spider):
    name = 'cars'

    def __init__(self, *args, **kwargs):
        # turn off annoying logging, set LOG_LEVEL=DEBUG in settings.py to see more logs
        super().__init__(*args, **kwargs)
        self.url = self.url


    def start_requests(self):
        yield scrapy.Request(url=self.url, callback=self.parse)


    def parse(self, response):
        # navigate to provided page
        for post in response.xpath(".//div/h4/span/a"):
            temp_post = post.xpath('.//@href').get()
            yield scrapy.Request(temp_post, self.parse_page_comments)

        next = False
        try:
            more = response.xpath(".//div/ul/li/a[contains(@rel, 'next')]/text()")
            if str(more.get()) == "Suivant":
                next = True
        except:
            pass

        if next == True:
            more_posts = response.xpath(".//div/ul/li/a[contains(@rel, 'next')]")
            more_posts = more_posts.xpath('.//@href').get()
            yield scrapy.Request(more_posts, self.parse_page_comments)
        else:
            print("no more pages")


    def parse_page_comments(self, response):
        texts = []
        dates = []
        for comment in response.xpath(".//div[contains(@data-role, 'commentContent')]"):
            if comment.get().find("dit") != -1:
                try:
                    review = cleanhtml(cleanresponse(str(comment.get()).replace(comment.xpath(".//blockquote").get(), ""))).\
                        replace("\t", ";").replace("\n", ";")
                    review = replace(review, ";")
                    texts.append(str(review))
                except:
                    texts.append("")
            else:
                review = cleanhtml(cleanresponse(comment.get())).replace("\t", ";").replace("\n", ";")
                review = replace(review, ";")
                texts.append(str(review))

        for date in response.xpath(".//div/div/a/time"):
            dates.append(date.get()[16:26])

        for i in range(len(dates)):
            new_review = Review()
            new_review['Text'] = str(texts[i])
            new_review['Date'] = str(dates[i])
            yield new_review


        try:
            more = response.xpath(".//div/ul/li/a[contains(@rel, 'next')]/text()")
            if str(more[-1].get()) == "Suivant":
                more_posts = response.xpath(".//div/ul/li/a[contains(@rel, 'next')]")[-1].xpath('.//@href').get()
                yield scrapy.Request(more_posts, callback=self.parse_page_comments)
        except:
            pass

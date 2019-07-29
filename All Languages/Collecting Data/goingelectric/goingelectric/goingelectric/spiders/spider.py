# -*- coding: utf-8 -*-
import scrapy
from scrapy.loader import ItemLoader
from scrapy.http import FormRequest
from goingelectric.items import Review
import re

def cleanresponse(raw_html):
    cleanr = re.compile('<div class="quotetitle">.*?</div>')
    cleantext = re.sub(cleanr, '.', raw_html)
    return cleantext


def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '.', raw_html)
    return cleantext

def replace(s, ch): # replace multiple occurrences of a character by a single character
    new_str = []
    l = len(s)
    for i in range(len(s)):
        if (s[i] == ch and i != (l - 1) and
                i != 0 and s[i + 1] != ch and s[i - 1] != ch):
            new_str.append(s[i])
        elif s[i] == ch:
            if ((i != (l - 1) and s[i + 1] == ch) and
                    (i != 0 and s[i - 1] != ch)):
                new_str.append(s[i])
        else:
            new_str.append(s[i])
    return ("".join(i for i in new_str))

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
        for post in response.xpath(".//a[contains(@class, 'topictitle')]"):
            temp_post = "https://www.goingelectric.de/forum" + post.xpath('.//@href').get()[1:]
            yield scrapy.Request(temp_post, self.parse_page_comments)

        next_page = response.xpath(".//a[contains(@class, 'button') and contains(@rel, 'next')]")
        next_page = "https://www.goingelectric.de/forum" + next_page.xpath('.//@href').get()[1:]
        if next_page:
            yield scrapy.Request(next_page, self.parse)
        else:
            print("no more pages")

    def parse_page_comments(self, response):
        texts = []
        dates = []
        IDs = []

        for id in response.xpath(".//div[contains(@id, 'post_content')]/@id"):
            IDs.append(id.get())

        for comment in response.xpath(".//div[contains(@class, 'content')]"):
            if comment.get().find("geschrieben") != -1:
                try:
                    review = cleanhtml(cleanresponse(str(comment.get()).replace(comment.xpath(".//blockquote").get(), ""))).\
                        replace("\t", ".").replace("\n", ".")
                    review = replace(review, ".")
                    texts.append(str(review))
                except:
                    texts.append("")
            else:
                review = cleanhtml(cleanresponse(comment.get())).replace("\t", ".").replace("\n", ".")
                review = replace(review, ".")
                texts.append(str(review))

        for date in response.xpath(".//dl[contains(@class, 'postprofile')]/ul/dd[contains(@class, 'profile-joined')]"):
            date = date.get().split(" ")
            if date[4][0:3] == "Jan":
                month = "01"
            elif date[4][0:3] == "Feb":
                month = "02"
            elif date[4][0:3] == "Mär":
                month = "03"
            elif date[4][0:3] == "Apr":
                month = "04"
            elif date[4][0:3] == "Mai":
                month = "05"
            elif date[4][0:3] == "Jun":
                month = "06"
            elif date[4][0:3] == "Jul":
                month = "07"
            elif date[4][0:3] == "Aug":
                month = "08"
            elif date[4][0:3] == "Sep":
                month = "09"
            elif date[4][0:3] == "Okt":
                month = "10"
            elif date[4][0:3] == "Nov":
                month = "11"
            elif date[4][0:3] == "Dez":
                month = "12"

            if date[3][1] == ".":
                day = "0" + date[3][0]
            else:
                day = date[3][0:2]

            try:
                date = date[5][0:4] + "-" + month + "-" + day
                dates.append(date)
            except:
                dates.append("2019-01-01")

        for i in range(len(dates)):
            new_review = Review()
            new_review['ID'] = str(IDs[i]).replace(" ","")
            new_review['Text'] = str(texts[i])
            new_review['Date'] = str(dates[i])
            yield new_review

        try:
            more = response.xpath(".//a[contains(@class, 'button') and contains(@rel, 'next')]")
            more_posts = "https://www.goingelectric.de/forum" + more.xpath('.//@href').get()[1:]
            yield scrapy.Request(more_posts, callback=self.parse_page_comments)
        except:
            pass
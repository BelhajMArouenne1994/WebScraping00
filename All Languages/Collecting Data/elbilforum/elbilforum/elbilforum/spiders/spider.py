# -*- coding: utf-8 -*-
import scrapy
from scrapy.loader import ItemLoader
from scrapy.http import FormRequest
from elbilforum.items import Review

import re
def cleanresponse(raw_html):
    cleanr = re.compile('<div class="quotetitle">.*?</div>')
    cleantext = re.sub(cleanr, ' ', raw_html)
    return cleantext


def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', raw_html)
    return cleantext


class CarsSpider(scrapy.Spider):
    name = 'cars'

    def start_requests(self):
        urls = ["https://elbilforum.no/index.php?board=47.0"]
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)


    def parse(self, response):
        # navigate to provided page
        for post in response.xpath(".//span/a[contains(@href, 'https://elbilforum.no/index.php?')]"):
            temp_post = post.xpath('.//@href').get()
            yield scrapy.Request(temp_post, self.parse_page_comments)

        response = str(response)[:-1]
        if str(response)[-2] == ".":
            page = int(int(str(response).split(".")[-1])+20)
            if page < 730:
                more_posts = str(response)[:-1] + str(page)
                more_posts = more_posts[5:]
                yield scrapy.Request(more_posts, self.parse_page_comments)
            else:
                print("no more pages")
        else:
            page = int(int(str(response).split(".")[-1])+20)
            if page < 730:
                more_posts = str(response)[:-2] + str(page)
                more_posts = more_posts[5:]
                yield scrapy.Request(more_posts, self.parse_page_comments)
            else:
                print("no more pages")

    def parse_page_comments(self, response):
        texts = []
        dates = []
        prev = ""
        for comment in response.xpath('.//div[contains(@class, "inner")]'):
            if comment.get().lower().find("sitat") != -1:
                try:
                    review = cleanhtml(cleanresponse(str(comment.xpath(".//quotefooter").get()))).\
                        replace("\t", ";").replace("\n", ";")
                    texts.append(str(review))
                except:
                    texts.append("")
            else:
                review = cleanhtml(cleanresponse(comment.get())).replace("\t", ";").replace("\n", ";")
                texts.append(str(review))


        for date in response.xpath('.//div[contains(@class, "keyinfo")]/div[contains(@class, "smalltext")]'):
            if date.get().find("Svar") == -1:
                date = date.get().split(" ")
                if date[6][0:3] == "Jan":
                    month = "01"
                elif date[6][0:3] == "Feb":
                    month = "02"
                elif date[6][0:3] == "Mar":
                    month = "03"
                elif date[6][0:3] == "Apr":
                    month = "04"
                elif date[6][0:3] == "Mai":
                    month = "05"
                elif date[6][0:3] == "Jun":
                    month = "06"
                elif date[6][0:3] == "Jul":
                    month = "07"
                elif date[6][0:3] == "Aug":
                    month = "08"
                elif date[6][0:3] == "Sep":
                    month = "09"
                elif date[6][0:3] == "Okt":
                    month = "10"
                elif date[6][0:3] == "Nov":
                    month = "11"
                elif date[6][0:3] == "Des":
                    month = "12"

                try:
                    date = date[7][0:4] + "-" + month + "-" + date[5][0:2]
                    dates.append(date)
                except:
                    dates.append("2019-01-01")
            else:
                date = date.get().split(" ")
                if date[7][0:3] == "Jan":
                    month = "01"
                elif date[7][0:3] == "Feb":
                    month = "02"
                elif date[7][0:3] == "Mar":
                    month = "03"
                elif date[7][0:3] == "Apr":
                    month = "04"
                elif date[7][0:3] == "Mai":
                    month = "05"
                elif date[7][0:3] == "Jun":
                    month = "06"
                elif date[7][0:3] == "Jul":
                    month = "07"
                elif date[7][0:3] == "Aug":
                    month = "08"
                elif date[7][0:3] == "Sep":
                    month = "09"
                elif date[7][0:3] == "Okt":
                    month = "10"
                elif date[7][0:3] == "Nov":
                    month = "11"
                elif date[7][0:3] == "Des":
                    month = "12"

                try:
                    date = date[8][0:4] + "-" + month + "-" + date[6][0:2]
                    dates.append(date)
                except:
                    dates.append("2019-01-01")

        for i in range(len(dates)):
            new_review = Review()
            new_review['Text'] = str(texts[i])
            new_review['Date'] = str(dates[i])
            yield new_review

        try:
            more = response.xpath('.//a[contains(@href, "next=next")]')
            more_posts = more.xpath('.//@href').get()
            yield scrapy.Request(more_posts, callback=self.parse_page_comments)
        except:
            pass

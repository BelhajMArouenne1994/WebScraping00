import hashlib
import re
import time

import scrapy
from automobile_propre.items import Review
from mtranslate import translate


def cleanresponse(raw_html):
    cleanr = re.compile('<div class="quotetitle">.*?</div>')
    cleantext = re.sub(cleanr, ' ', raw_html)
    return cleantext


def replace_multiple(ch, s):
    new_str = ''.join([ch[i] for i in range(len(ch)-1) if (ch[i+1] != s or ch[i] != s)]+[ch[-1]])
    return new_str


def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '.', raw_html)
    cleantext = cleantext.replace("\r", ".").replace("\n", ".").replace("\t", ".").replace(";", ".")
    cleantext = replace_multiple(cleantext, " ")
    cleantext = replace_multiple(cleantext, ".")
    return cleantext


def replace_duplicate_car(s, ch):
    l = len(s)
    bool = True
    while bool == True:
        new_str = []
        c = 0
        for i in range(len(s)):
            if s[i] == ch and i < (l-2) and i != 0 and (s[i+1] != ch or s[i+2] != ch):
                c += 1
            else:
                new_str.append(s[i])
        if c != 0:
            bool = True
        else:
            bool = False
        s = "".join(i for i in new_str)
        l = len(s)
    return "".join(i for i in new_str)


class CarsSpider(scrapy.Spider):
    name = 'cars'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.link = self.link
        self.car = self.car

    def start_requests(self):
        yield scrapy.Request(url=self.link, callback=self.parse)

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
                    review = cleanhtml(
                        cleanresponse(str(comment.get()).replace(comment.xpath(".//blockquote").get(), ""))). \
                        replace("\t", ";").replace("\n", ";")
                    review = replace_duplicate_car(review, ";")
                    texts.append(str(review))
                except:
                    texts.append("")
            else:
                review = cleanhtml(cleanresponse(comment.get())).replace("\t", ";").replace("\n", ";")
                review = replace_duplicate_car(review, ";")
                texts.append(str(review))

        for date in response.xpath(".//div/div/a/time"):
            dates.append(date.get()[16:26])

        for i in range(len(dates)):
            if texts[i] != "" and texts[i] != " ":
                new_review = Review()
                new_review['ID'] = str(hashlib.md5(texts[i].encode('utf-8')).hexdigest())
                try:
                    new_review['Text'] = translate(str(texts[i]), "en", "fr")
                except:
                    print("wainting")
                    time.sleep(1)
                new_review['Date'] = str(dates[i])
                new_review['car'] = self.car
                new_review['Country'] = "France"
                new_review['Page_Type'] = "Forum"
                yield new_review

        try:
            more = response.xpath(".//div/ul/li/a[contains(@rel, 'next')]/text()")
            if str(more[-1].get()) == "Suivant":
                more_posts = response.xpath(".//div/ul/li/a[contains(@rel, 'next')]")[-1].xpath('.//@href').get()
                yield scrapy.Request(more_posts, callback=self.parse_page_comments)
        except:
            pass

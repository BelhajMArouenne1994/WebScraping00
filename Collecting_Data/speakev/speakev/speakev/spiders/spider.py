import re

import scrapy
from speakev.items import Review


def cleanresponse(raw_html):
    cleanr = re.compile('<div class="quotetitle">.*?</div>')
    cleantext = re.sub(cleanr, '.', raw_html)
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


def replace(s, ch):  # replace multiple occurrences of a character by a single character
    new_str = []
    l = len(s)
    for i in range(len(s)):
        if (s[i] == ch and i != (l-1) and
                i != 0 and s[i+1] != ch and s[i-1] != ch):
            new_str.append(s[i])
        elif s[i] == ch:
            if ((i != (l-1) and s[i+1] == ch) and
                    (i != 0 and s[i-1] != ch)):
                new_str.append(s[i])
        else:
            new_str.append(s[i])
    return "".join(i for i in new_str)


class CarsSpider(scrapy.Spider):
    name = 'cars'

    def __init__(self, *args, **kwargs):
        # turn off annoying logging, set LOG_LEVEL=DEBUG in settings.py to see more logs
        super().__init__(*args, **kwargs)
        self.url = self.url
        self.car = self.car

    def start_requests(self):
        yield scrapy.Request(url=self.url, callback=self.parse)

    def parse(self, response):
        # navigate to provided page
        print(response)
        for post in response.xpath(".//a[contains(@href, 'thread')]"):
            temp_post = "https://www.speakev.com/"+post.xpath('.//@href').get()[1:]
            yield scrapy.Request(temp_post, self.parse_page_comments)

        next_page = response.xpath(".//a[contains(@class, 'next')]")
        if next_page:
            next_page = "https://www.speakev.com/"+next_page.xpath('.//@href').get()[1][1:]
            yield scrapy.Request(next_page, self.parse)
        else:
            print("no more pages")

    def parse_page_comments(self, response):
        texts = []
        dates = []
        IDs = []

        for id in response.xpath(".//article[contains(@data-content, 'post')]/@id"):
            IDs.append(id.get())

        for comment in response.xpath(
                ".//article[contains(@class, 'message-body')]/div[contains(@class, 'bbWrapper')]"):
            review = cleanhtml(cleanresponse(comment.get())).replace("\t", ".").replace("\n", ".")
            review = replace(review, ".")
            texts.append(str(review))

        for date in response.xpath(".//div[contains(@class, 'message-main')]/header/div/a/time/@datetime"):
            date = date.get().split("T")
            dates.append(date[0])

        for i in range(len(dates)):
            if texts[i] != "" and texts[i] != " ":
                new_review = Review()
                new_review['ID'] = str(IDs[i]).replace(" ", "")
                new_review['Text'] = str(texts[i])
                new_review['Date'] = str(dates[i])
                new_review['Car'] = self.car
                new_review['Country'] = "ALL"
                new_review['Page_Type'] = "Forum"
                yield new_review

        try:
            more = response.xpath(".//a[contains(@class, 'next')]")
            more_posts = "https://www.speakev.com/"+more.xpath('.//@href').get()[1:]
            yield scrapy.Request(more_posts, callback=self.parse_page_comments)
        except:
            pass

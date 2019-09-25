import re
import time

import scrapy
from Facebook.items import FbcrawlItem, Post
from mtranslate import translate
from scrapy.http import FormRequest
from scrapy.loader import ItemLoader


def find_between(s, first, last):
    try:
        start = s.rindex(first)+len(first)
        end = s.rindex(last, start)
        return s[start:end]
    except ValueError:
        return ""


def replace_multiple(ch, s):
    new_str = ''.join([ch[i] for i in range(len(ch)-1) if (ch[i+1] != s or ch[i] != s)]+[ch[-1]])
    return new_str


def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '.', raw_html)
    cleanr2 = re.compile(("J'aime*?"))
    cleantext = re.sub(cleanr2, '', cleantext)
    cleantext = cleantext.replace("\r", ".").replace("\n", ".").replace("\t", ".").replace(";", ".")
    cleantext = replace_multiple(cleantext, " ")
    cleantext = replace_multiple(cleantext, ".")
    return cleantext


class FacebookSpider(scrapy.Spider, ):
    name = "cars"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.email = self.email
        self.password = self.password
        self.url = self.url
        self.language = self.language
        self.country = self.country
        self.car = self.car

        self.lang = 'en'

        # current year, this variable is needed for parse_page recursion
        self.k = int(time.strftime("%Y"))

        # count number of posts, used to prioritized parsing and correctly insert in the csv
        self.count = 0

        self.start_urls = ['https://mbasic.facebook.com']

        self.exit = 0

    def parse(self, response):
        if self.exit != 1:
            return FormRequest.from_response(
                response,
                formxpath='//form[contains(@action, "login")]',
                formdata={'email': self.email, 'pass': self.password},
                callback=self.parse_home
            )

    def parse_home(self, response):
        # handle 'save-device' redirection
        if response.xpath("//div/a[contains(@href,'save-device')]"):
            return FormRequest.from_response(
                response,
                formdata={'name_action_selected': 'dont_save'},
                callback=self.parse_home
            )

        # set language interface
        if self.lang == '_':
            if response.xpath("//input[@placeholder='Search Facebook']"):
                self.logger.info('Language recognized: lang="en"')
                self.lang = 'en'
            else:
                raise AttributeError('Language not recognized\n'
                                     'Change your interface lang from facebook '
                                     'and try again')

        # navigate to provided page
        href = self.url
        return scrapy.Request(url=href, callback=self.parse_page, meta={'index': 1})

    def parse_page(self, response):
        '''
        Parse the given page selecting the posts.
        Then ask recursively for another page.
        '''
        # select all posts
        for post in response.xpath(".//a[contains(@href,'story')]"):
            if ((post.get()[-12:]) == "Comments</a>"):
                new = ItemLoader(item=FbcrawlItem(), selector=post.xpath('.//@href'))
                self.logger.info('Parsing post n = {}'.format(abs(self.count)))

                # page_url #new.add_value('url',response.url)
                # returns full post-link in a list
                temp_post = 'https://mbasic.facebook.com'+post.xpath('.//@href').get()
                self.count -= 1
                yield scrapy.Request(temp_post, callback=self.parse_page_comments, priority=self.count,
                                     meta={'item': new})

        mores = response.xpath(".//a[contains(@href, 'multi_permalinks')]")
        for more in mores:
            if more and str(more.xpath('.//@href').get()).lower().find("see more posts") and \
                    str(more.xpath('.//@href').get()).lower()[0:6] == "/group":
                more_posts = "https://mbasic.facebook.com"+more.xpath('.//@href').get()
                print(more_posts)
                # load following page
                # tries to click on "more", otherwise it looks for the appropriate
                # year for 1-click only and proceeds to click on others
                yield scrapy.Request(more_posts, callback=self.parse_page, meta={'flag': self.k})
            else:
                print("no more pages")

    def parse_page_comments(self, response):
        new_post = ItemLoader(item=Post(), response=response)
        ID = response.xpath('.//div[contains(@data-ft,"top_level")]/@data-ft').get()[21:39]

        new_post.add_value('ID', str(ID))
        new_post.add_xpath('src',
                           "//td/div/h3/strong/a/text() | //span/strong/a/text() | //div/div/div/a[contains(@href,'post_id')]/strong/text()")
        new_post.add_xpath('Date', '//div/div/abbr/text()')

        text_post = response.xpath('//div[@data-ft]//p//text() | //div[@data-ft]/div[@class]/div[@class]/text()').get()
        text_post = cleanhtml(text_post)

        translated = False
        while translated == False:
            try:
                text_post = translate(str(text_post), "en", self.language)
                translated = True
            except:
                print("waiting")
                time.sleep(1)

        new_post.add_value('Text', text_post)
        new_post.add_value('Car', self.car)
        new_post.add_value('Language', self.language)
        new_post.add_value('Country', self.country)
        new_post.add_value('Page_Type', "Facebook")
        yield new_post.load_item()

        path2 = './/div[string-length(@class) = 2 and count(@id)=1 and contains("0123456789", substring(@id,1,1)) and not(.//div[contains(@id,"comment_replies")])]'
        for i, reply in enumerate(response.xpath(path2)):

            text = reply.xpath('.//div[h3]/div[1]/text()').get()
            text = cleanhtml(text)

            ID_2 = str(ID)+"_"+str(text)

            href = reply.xpath('.//div[h3]/div[1]/@href').extract()
            stop_comments = ["https://", "NONE", ".", " .", " .", " . ", ""]

            if (text in stop_comments) or (len(href) > 0):
                pass

            else:
                new = ItemLoader(item=FbcrawlItem(), selector=reply)
                new.add_value('ID', str(ID_2))

                translated = False
                while translated == False:
                    try:
                        text = translate(str(text), "en", self.language)
                        translated = True
                    except:
                        print("waiting")
                        time.sleep(1)

                new.add_value('Text', text)
                new.add_xpath('Date', './/abbr/text()')
                new.add_xpath('likes', './/a[contains(@href,"reaction/profile")]//text()')
                new.add_value('post_url', str(response.url))
                new.add_value('Car', self.car)
                new.add_value('Language', self.language)
                new.add_value('Country', self.country)
                new.add_value('Page_Type', "Facebook")
                yield new.load_item()

        try:
            for next_page in response.xpath('.//div[contains(@id,"see_next")]'):
                new_page = next_page.xpath('.//@href').extract()
                new_page = response.urljoin(new_page[0])
                yield scrapy.Request(new_page,
                                     callback=self.parse_page_comments,
                                     meta={'index': 1})
        except:
            pass

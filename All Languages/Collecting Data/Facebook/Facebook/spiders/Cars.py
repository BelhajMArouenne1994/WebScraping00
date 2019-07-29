import scrapy
import logging
import urllib
from requests import HTTPError
import os, requests, uuid, json

from scrapy.loader import ItemLoader
from scrapy.http import FormRequest
from Facebook.items import FbcrawlItem, Post

import re
import time


########################################################################################

Page = ["Renault Zoe Owners Group UK",
        "Renault Zoe - Italia e Svizzera",
        "Official Renault Zoe & Z.E. Owners Club (RZOC)",
        "Renault ZOE Francophone",
        "Renault Zoe - Hong Kong",
        "ZE Renault ZOE",
        "Renault Zoe Türkiye",
        "Renault Zoe Klub Magyarország"]

URL = ["https://mbasic.facebook.com/groups/953405434715346?refid=46&__xts__%5B0%5D=12.%7B%22unit_id_click_type%22%3A%22graph_search_results_item_in_module_tapped%22%2C%22click_type%22%3A%22result%22%2C%22module_id%22%3A3%2C%22result_id%22%3A953405434715346%2C%22session_id%22%3A%223554c15b3317e4bdcb993dbfbad67d87%22%2C%22module_role%22%3A%22ENTITY_GROUPS%22%2C%22unit_id%22%3A%22browse_rl%3Abda687ef-345c-4f43-b790-83558433ccba%22%2C%22browse_result_type%22%3A%22browse_type_group%22%2C%22unit_id_result_id%22%3A953405434715346%2C%22module_result_position%22%3A0%7D",
       "https://mbasic.facebook.com/groups/542911602465223?refid=46&__xts__%5B0%5D=12.%7B%22unit_id_click_type%22%3A%22graph_search_results_item_tapped%22%2C%22click_type%22%3A%22result%22%2C%22module_id%22%3A1%2C%22result_id%22%3A542911602465223%2C%22session_id%22%3A%22631aa99c7142a195ae9111ecce97ffe9%22%2C%22module_role%22%3A%22ENTITY_GROUPS%22%2C%22unit_id%22%3A%22browse_rl%3A9e871b80-dbef-4503-b4ff-0794b6c75d11%22%2C%22browse_result_type%22%3A%22browse_type_group%22%2C%22unit_id_result_id%22%3A542911602465223%2C%22module_result_position%22%3A0%7D",
       "https://mbasic.facebook.com/groups/706135872863284?refid=46&__xts__%5B0%5D=12.%7B%22unit_id_click_type%22%3A%22graph_search_results_item_tapped%22%2C%22click_type%22%3A%22result%22%2C%22module_id%22%3A1%2C%22result_id%22%3A706135872863284%2C%22session_id%22%3A%224cdbd3003d9622e2cad3d015d349fb26%22%2C%22module_role%22%3A%22ENTITY_GROUPS%22%2C%22unit_id%22%3A%22browse_rl%3A7aeb9a9b-ca21-4b1e-9b24-e79d8ac33947%22%2C%22browse_result_type%22%3A%22browse_type_group%22%2C%22unit_id_result_id%22%3A706135872863284%2C%22module_result_position%22%3A0%7D",
       "https://mbasic.facebook.com/groups/618220815041945?refid=46&__xts__%5B0%5D=12.%7B%22unit_id_click_type%22%3A%22graph_search_results_item_tapped%22%2C%22click_type%22%3A%22result%22%2C%22module_id%22%3A1%2C%22result_id%22%3A618220815041945%2C%22session_id%22%3A%222911e8410831cf0319b20dbd689cf60d%22%2C%22module_role%22%3A%22ENTITY_GROUPS%22%2C%22unit_id%22%3A%22browse_rl%3A8e1e7951-d64f-4063-9ba3-492165d2ff29%22%2C%22browse_result_type%22%3A%22browse_type_group%22%2C%22unit_id_result_id%22%3A618220815041945%2C%22module_result_position%22%3A0%7D",
       "https://mbasic.facebook.com/groups/500084667012780?refid=46&__xts__%5B0%5D=12.%7B%22unit_id_click_type%22%3A%22graph_search_results_item_tapped%22%2C%22click_type%22%3A%22result%22%2C%22module_id%22%3A1%2C%22result_id%22%3A500084667012780%2C%22session_id%22%3A%221306b4269619608c5971a00963974ade%22%2C%22module_role%22%3A%22ENTITY_GROUPS%22%2C%22unit_id%22%3A%22browse_rl%3A3ae07c2f-7663-42f9-a344-468c5a1a3a83%22%2C%22browse_result_type%22%3A%22browse_type_group%22%2C%22unit_id_result_id%22%3A500084667012780%2C%22module_result_position%22%3A0%7D",
       "https://mbasic.facebook.com/groups/465117457211621?refid=46&__xts__%5B0%5D=12.%7B%22unit_id_click_type%22%3A%22graph_search_results_item_tapped%22%2C%22click_type%22%3A%22result%22%2C%22module_id%22%3A1%2C%22result_id%22%3A465117457211621%2C%22session_id%22%3A%22b1778eac31b2fee6c25df464dad6f9b5%22%2C%22module_role%22%3A%22ENTITY_GROUPS%22%2C%22unit_id%22%3A%22browse_rl%3A54e0ca00-9c07-405d-a427-df5e16ce8dce%22%2C%22browse_result_type%22%3A%22browse_type_group%22%2C%22unit_id_result_id%22%3A465117457211621%2C%22module_result_position%22%3A0%7D",
       "https://mbasic.facebook.com/groups/171681613407666?refid=46&__xts__%5B0%5D=12.%7B%22unit_id_click_type%22%3A%22graph_search_results_item_tapped%22%2C%22click_type%22%3A%22result%22%2C%22module_id%22%3A1%2C%22result_id%22%3A171681613407666%2C%22session_id%22%3A%22cdbd26f2f104d163ce64b3bfca44a030%22%2C%22module_role%22%3A%22ENTITY_GROUPS%22%2C%22unit_id%22%3A%22browse_rl%3A26150a89-8b56-4898-bc65-10e356ecd8a7%22%2C%22browse_result_type%22%3A%22browse_type_group%22%2C%22unit_id_result_id%22%3A171681613407666%2C%22module_result_position%22%3A0%7D",
       "https://mbasic.facebook.com/groups/663700640660894?refid=46&__xts__%5B0%5D=12.%7B%22unit_id_click_type%22%3A%22graph_search_results_item_tapped%22%2C%22click_type%22%3A%22result%22%2C%22module_id%22%3A1%2C%22result_id%22%3A663700640660894%2C%22session_id%22%3A%2237ea54a63944306a686b82de03ea6261%22%2C%22module_role%22%3A%22ENTITY_GROUPS%22%2C%22unit_id%22%3A%22browse_rl%3A2a8d694e-e151-48cb-b1d0-4d1af235e8ad%22%2C%22browse_result_type%22%3A%22browse_type_group%22%2C%22unit_id_result_id%22%3A663700640660894%2C%22module_result_position%22%3A0%7D"]

Language = ["en",
            "it",
            "en",
            "fr",
            "zh",
            "es",
            "tr",
            "hu"]

Collection = ["ZOE_en",
              "ZOE_it",
              "ZOE_en",
              "ZOE_fr",
              "ZOE_zh",
              "ZOE_es",
              "ZOE_tr",
              "ZOE_hu"]

#scrapy crawl cars - arguments ()

########################################################################################


def find_between(s, first, last):
    try:
        start = s.rindex(first) + len(first)
        end = s.rindex(last, start)
        return s[start:end]
    except ValueError:
        return ""


def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    cleanr2 = re.compile(("J'aime*?"))
    cleantext2 = re.sub(cleanr2, '', cleantext)
    return cleantext2


class FacebookSpider(scrapy.Spider):
    """
    Parse FB pages (needs credentials)
    """
    name = "cars"

    def __init__(self, *args, **kwargs):
        # turn off annoying logging, set LOG_LEVEL=DEBUG in settings.py to see more logs
        super().__init__(*args, **kwargs)

        self.group = self.group

        positions = [i for i, x in enumerate(Page) if x == self.group]
        if len(positions) != 0:
            self.index = positions[0]
            self.url = URL[self.index]
            self.language = Language[self.index]
            self.collection = Collection[self.index]

            self.email = "belhajbenky@yahoo.fr"
            self.password = "ridhabelhaj"

            self.lang = 'en'

            # current year, this variable is needed for parse_page recursion
            self.k = int(time.strftime("%Y"))

            # count number of posts, used to prioritized parsing and correctly insert in the csv
            self.count = 0

            self.start_urls = ['https://mbasic.facebook.com']

            self.exit = 0
        else:
            print("Page facebook pas prise en consideration")

            print("Rajouter ces informations au fichier (Cars.py): \n"
                  "*********************************************** \n"
                   "Page (nom de la page) (ligne 17) \n"
                   "URL (ligne 26) \n"
                   "Language (ligne 35) \n"
                   "Collection (nom de la base de données (car-lang))(ligne 44) \n"
                  "*********************************************** \n")
            self.exit = 1


    def parse(self, response):
        if self.exit != 1:
            return FormRequest.from_response(
                response,
                formxpath='//form[contains(@action, "login")]',
                formdata={'email': self.email, 'pass': self.password},
                callback=self.parse_home
            )

    def parse_home(self, response):
        '''
        This method has multiple purposes:
        1) Handle failed logins due to facebook 'save-device' redirection
        2) Set language interface, if not already provided
        3) Navigate to given page
        '''
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
        href = URL[self.index]
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
                temp_post = 'https://mbasic.facebook.com' + post.xpath('.//@href').get()
                self.count -= 1
                yield scrapy.Request(temp_post, callback=self.parse_page_comments, priority=self.count, meta={'item': new})

        mores = response.xpath(".//a[contains(@href, 'multi_permalinks')]")
        for more in mores:
            if more and str(more.xpath('.//@href').get()).lower().find("see more posts") and \
                    str(more.xpath('.//@href').get()).lower()[0:6] == "/group":
                more_posts = "https://mbasic.facebook.com" + more.xpath('.//@href').get()
                print(more_posts)
                # load following page
                # tries to click on "more", otherwise it looks for the appropriate
                # year for 1-click only and proceeds to click on others
                yield scrapy.Request(more_posts, callback=self.parse_page, meta={'flag': self.k})
            else:
                print("no more pages")

    def parse_page_comments(self, response):
        new_post = ItemLoader(item=Post(), response=response)
        new_post.add_xpath('src',
                           "//td/div/h3/strong/a/text() | //span/strong/a/text() | //div/div/div/a[contains(@href,'post_id')]/strong/text()")
        new_post.add_xpath('date', '//div/div/abbr/text()')
        text_post = response.xpath('//div[@data-ft]//p//text() | //div[@data-ft]/div[@class]/div[@class]/text()').get()

        new_post.add_value('text', str(text_post))
        yield new_post.load_item()

        path2 = './/div[string-length(@class) = 2 and count(@id)=1 and contains("0123456789", substring(@id,1,1)) and not(.//div[contains(@id,"comment_replies")])]'
        for i, reply in enumerate(response.xpath(path2)):
            text = reply.xpath('.//div[h3]/div[1]/text()').get()

            href = reply.xpath('.//div[h3]/div[1]/@href').extract()
            stop_comments = ["https://", "NONE", ".", " .", " .", " . ", ""]
            if (text in stop_comments) or (len(href) > 0):
                pass
            else:

                new = ItemLoader(item=FbcrawlItem(), selector=reply)
                new.context['lang'] = self.lang
                new.add_value('text', str(text))
                new.add_xpath('date', './/abbr/text()')
                new.add_xpath('likes', './/a[contains(@href,"reaction/profile")]//text()')
                new.add_value('post_url', str(response.url))
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


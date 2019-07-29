# -*- coding: utf-8 -*-
from scrapy.exceptions import DropItem
from scrapy.conf import settings
import logging
import pymongo
from goingelectric.items import Review


class SaveToMongoPipeline(object):

    ''' pipeline that save data to mongodb '''
    def __init__(self):
        connection = pymongo.MongoClient('mongodb://127.0.0.1:27017/?gssapiServiceName=mongodb')
        db = connection[settings['MONGODB_DB']]
        self.commentCollection = db[settings['MONGODB_REVIEWS_COLLECTION']]
        self.commentCollection.ensure_index([('ID', pymongo.ASCENDING)], unique=True, dropDups=True)

    def process_item(self, item, spider):
        if isinstance(item, Review):
            dbItem = self.commentCollection.find_one({'ID': item['ID']})
            if dbItem:
                pass
            else:
                self.commentCollection.insert_one(dict(item))

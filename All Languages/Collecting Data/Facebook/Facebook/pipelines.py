# -*- coding: utf-8 -*-
from scrapy.exceptions import DropItem
from scrapy.conf import settings
import logging
import pymongo
from Facebook.items import FbcrawlItem, Post

logger = logging.getLogger(__name__)

class SaveToMongoPipeline(object):

    ''' pipeline that save data to mongodb '''
    def __init__(self):
        connection = pymongo.MongoClient('mongodb://127.0.0.1:27017/?gssapiServiceName=mongodb')
        db = connection[settings['MONGODB_DB']]
        self.commentCollection = db[settings['MONGODB_FBCRAWLITEM_COLLECTION']]
        self.postCollection = db[settings['MONGODB_POST_COLLECTION']]
        #self.commentCollection.ensure_index([('ID', pymongo.ASCENDING)], unique=True, dropDups=True)
        #self.postCollection.ensure_index([('src', pymongo.ASCENDING)], unique=True, dropDups=True)

    def process_item(self, item, spider):
        if isinstance(item, FbcrawlItem):
            self.commentCollection.insert_one(dict(item))

        elif isinstance(item, Post):
            self.postCollection.insert_one(dict(item))

        else:
            logger.info("Item type is not recognized! type = %s" %type(item))
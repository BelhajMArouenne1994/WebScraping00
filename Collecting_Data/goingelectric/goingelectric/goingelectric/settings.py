BOT_NAME = 'goingelectric'

SPIDER_MODULES = ['goingelectric.spiders']
NEWSPIDER_MODULE = 'goingelectric.spiders'

# The amount of time (in secs) that the downloader should wait before downloading consecutive pages from the same website.
DOWNLOAD_DELAY = 0.25  # 0.25 s of delay

# Retry many times since proxies often fail
RETRY_TIMES = 10
# Retry on most error codes since proxies fail for different reasons
RETRY_HTTP_CODES = [500, 503, 504, 400, 403, 404, 408]

DOWNLOADER_MIDDLEWARES = {
    'scrapy.downloadermiddlewares.retry.RetryMiddleware': 90,
    # 'rotating_proxies.middlewares.RotatingProxyMiddleware': 610,
    # 'rotating_proxies.middlewares.BanDetectionMiddleware': 620,
    'scrapy.downloadermiddlewares.httpproxy.HttpProxyMiddleware': 110,
    'scrapy.downloadermiddlewares.useragent.UserAgentMiddleware': None,
    # 'scrapy_user_agents.middlewares.RandomUserAgentMiddleware': 400,
}

# Crawl responsibly by identifying yourself (and your website) on the user-agent
USER_AGENT_LIST = 'C:/Users/p100623/PycharmProjects/WebScraping/All Languages/Collecting_Data/goingelectric/goingelectric/useragents.txt'
PROXY_LIST = 'C:/Users/p100623/PycharmProjects/WebScraping/All Languages/Collecting_Data/goingelectric/goingelectric/proxy-list.txt'

# Proxy mode
# 0 = Every requests have different proxy
# 1 = Take only one proxy from the list and assign it to every requests
# 2 = Put a custom proxy to use in the settings
PROXY_MODE = 0

ITEM_PIPELINES = {
    'goingelectric.pipelines.SaveToMongoPipeline': 100,
}

# settings for mongodbgoingelectric
MONGODB_USER = ""
MONGODB_PASSWORD = ""
MONGODB_SERVER = "127.0.0.1"
MONGODB_PORT = 27017
MONGODB_DB = "Renault"  # database name to save the crawled data
MONGODB_REVIEWS_COLLECTION = "cars"  # collection name to save comments

# Obey robots.txt rules
ROBOTSTXT_OBEY = False

# Configure maximum concurrent requests performed by Scrapy (default: 16)
CONCURRENT_REQUESTS = 1

FEED_EXPORT_ENCODING = 'utf-8'
DUPEFILTER_DEBUG = True
LOG_LEVEL = 'INFO'

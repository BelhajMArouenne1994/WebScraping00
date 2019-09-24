import scrapy

class Review(scrapy.Item):
    ID = scrapy.Field()
    Text = scrapy.Field()
    Date = scrapy.Field()
    Car = scrapy.Field()
    Country = scrapy.Field()
    Page_Type = scrapy.Field()
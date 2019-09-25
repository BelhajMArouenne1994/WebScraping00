from datetime import datetime, timedelta

import scrapy
from scrapy.loader.processors import TakeFirst, Join


def parse_date(init_date, loader_context):
    lang = "en"

    from datetime import date
    today = date.today()
    today_date = today.strftime("%d-%m-%Y")
    # =============================================================================
    # English - status:beta
    # =============================================================================
    if lang == 'en':
        months = {
            'january': 1,
            'february': 2,
            'march': 3,
            'april': 4,
            'may': 5,
            'june': 6,
            'july': 7,
            'august': 8,
            'september': 9,
            'october': 10,
            'november': 11,
            'december': 12
        }

        months_abbr = {
            'jan': 1,
            'feb': 2,
            'mar': 3,
            'apr': 4,
            'may': 5,
            'jun': 6,
            'jul': 7,
            'aug': 8,
            'sep': 9,
            'oct': 10,
            'nov': 11,
            'dec': 12
        }

        date = init_date[0].split()
        year, month, day = [int(i) for i in str(datetime.now().date()).split(sep='-')]  # default is today
        l = len(date)

        # sanity check
        if l == 0:
            return 'Error: no data'
        # Yesterday, Now, 4hr, 50mins
        elif l == 1:
            if date[0].isalpha():
                if date[0].lower() == 'yesterday':
                    day = int(str(datetime.now().date()-timedelta(1)).split(sep='-')[2])
                    # check that yesterday was not in another month
                    month = int(str(datetime.now().date()-timedelta(1)).split(sep='-')[1])
                elif date[0].lower() == 'now':
                    return str(datetime(int(year), month, day).date())  # return today
                else:  # not recognized, (return date or init_date)
                    return str(date)
            else:
                # 4h, 50min (exploit future parsing)
                l = 2
                new_date = [x for x in date[0] if x.isdigit()]
                date[0] = ''.join(new_date)
                new_date = [x for x in date[0] if not (x.isdigit())]
                date[1] = ''.join(new_date)
        # l = 2
        elif l == 2:
            # 22 min
            if date[1] == 'min' or date[1] == 'mins':
                if int(str(datetime.now().time()).split(sep=':')[1])-int(date[0]) >= 0:
                    return str(datetime(int(year), month, day).date())
                # 22 min
                else:
                    day = int(str(datetime.now().date()-timedelta(1)).split(sep='-')[2])
                    month = int(str(datetime.now().date()-timedelta(1)).split(sep='-')[1])
                    return str(datetime(int(year), month, day).date())
            # 4 h
            elif date[1] == 'hr' or date[1] == 'hrs':
                if int(str(datetime.now().time()).split(sep=':')[0])-int(date[0]) >= 0:
                    return str(datetime(int(year), month, day).date())
                # 4 h
                else:
                    day = int(str(datetime.now().date()-timedelta(1)).split(sep='-')[2])
                    month = int(str(datetime.now().date()-timedelta(1)).split(sep='-')[1])
                    return str(datetime(int(year), month, day).date())
            # Feb 23
            elif len(date[0]) == 3 and date[0].isalpha():
                day = int(date[1])
                month = months_abbr[date[0].lower()]
                return str(datetime(int(year), month, day).date())
            # February 23
            elif len(date[0]) > 3 and date[0].isalpha():
                day = int(date[1])
                month = months[date[0]]
                return str(datetime(int(year), month, day).date())
            # parsing failed
            else:
                return str(date)

        # l = 3
        elif l == 3:
            # Jun 21 2017
            if len(date[0]) == 3 and date[2].isdigit():
                day = int(date[1].replace(",", ""))
                month = months_abbr[date[0].lower()]
                year = int(date[2])
                return str(datetime(year, month, day).date())
            # June 21 2017
            elif len(date[0]) > 3 and date[2].isdigit():
                day = int(date[1].replace(",", ""))
                month = months[date[0].lower()]
                year = int(date[2])
                return str(datetime(year, month, day).date())
            # parsing failed
            else:
                return today_date

        # l = 4
        elif l == 4:
            if date[0].lower() == "today":
                return datetime.now().date()
            elif (date[0].lower() == 'yesterday' and date[1] == 'at'):
                day = int(str(datetime.now().date()-timedelta(1)).split(sep='-')[2])
                month = int(str(datetime.now().date()-timedelta(1)).split(sep='-')[1])
                return str(datetime(int(year), int(month), int(day)).date())
            else:
                return str(date)

        # l = 5
        elif l == 5:
            if date[2] == 'at':
                # Jan 29 at 10:00 PM
                if len(date[0]) == 3:
                    day = int(date[1])
                    month = months_abbr[date[0].lower()]
                    return str(datetime(int(year), month, day).date())

                # 29 febbraio alle ore 21:49
                else:
                    day = int(date[1])
                    month = months[date[0].lower()]
                    return str(datetime(int(year), month, day).date())
            # parsing failed
            else:
                return str(date)

        # l = 6
        elif l == 6:
            if date[3] == 'at':
                date[1]
                # Aug 25, 2016 at 7:00 PM
                if len(date[0]) == 3:
                    day = int(date[1][:-1])
                    month = months_abbr[date[0].lower()]
                    year = int(date[2])
                    return str(datetime(int(year), month, day).date())
                # August 25, 2016 at 7:00 PM
                else:
                    day = int(date[1][:-1])
                    month = months[date[0].lower()]
                    year = int(date[2])
                    return str(datetime(int(year), month, day).date())
            # parsing failed
            else:
                return str(date)

        # l > 6
        # parsing failed - l too big
        else:
            return str(date)
    # parsing failed - language not supported
    else:
        return init_date


def comments_strip(string, loader_context):
    new_string = string[0].rstrip(' Comments')
    while new_string.rfind(',') != -1:
        new_string = new_string[0:new_string.rfind(',')]+new_string[new_string.rfind(',')+1:]
    return new_string


def reactions_strip(string, loader_context):
    newstring = string[0]
    # 19,298,873
    if len(newstring.split()) == 1:
        while newstring.rfind(',') != -1:
            newstring = newstring[0:newstring.rfind(',')]+newstring[newstring.rfind(',')+1:]
        return newstring
    else:
        return newstring


def url_strip(url):
    fullurl = url[0]
    i = fullurl.find('&id=')
    if i != -1:
        return fullurl[:i+4]+fullurl[i+4:].split('&')[0]
    else:  # catch photos
        i = fullurl.find('/photos/')
        if i != -1:
            return fullurl[:i+8]+fullurl[i+8:].split('/?')[0]
        else:  # catch albums
            i = fullurl.find('/albums/')
            if i != -1:
                return fullurl[:i+8]+fullurl[i+8:].split('/?')[0]
            else:
                return fullurl


class FbcrawlItem(scrapy.Item):
    ID = scrapy.Field()
    Date = scrapy.Field(  # when was the post published
        input_processor=TakeFirst(),
        output_processor=parse_date
    )
    Text = scrapy.Field(
        output_processor=Join(separator=u'')
    )  # full text of the post
    post_url = scrapy.Field(
        output_processor=Join(separator=u'')
    )  # user id
    username = scrapy.Field(
        output_processor=Join(separator=u'')
    )  # username of post
    likes = scrapy.Field(
        output_processor=reactions_strip
    )
    share = scrapy.Field(
        output_processor=Join(separator=u'')
    )
    Car = scrapy.Field(
        output_processor=Join(separator=u'')
    )
    Language = scrapy.Field(
        output_processor=Join(separator=u'')
    )
    Country = scrapy.Field(
        output_processor=Join(separator=u'')
    )
    Page_Type = scrapy.Field(
        output_processor=Join(separator=u'')
    )


class Post(scrapy.Item):
    ID = scrapy.Field(
        output_processor=Join(separator=u'')
    )
    src = scrapy.Field(

    )
    Date = scrapy.Field(  # when was the post published
        input_processor=TakeFirst(),
        output_processor=parse_date
    )
    Text = scrapy.Field(
        output_processor=Join(separator=u'')
    )  # full text of the post
    Car = scrapy.Field(
        output_processor=Join(separator=u'')
    )
    Language = scrapy.Field(
        output_processor=Join(separator=u'')
    )
    Country = scrapy.Field(
        output_processor=Join(separator=u'')
    )
    Page_Type = scrapy.Field(
        output_processor=Join(separator=u'')
    )

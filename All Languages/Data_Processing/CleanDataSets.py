import preprocessor as p
import math
import re
import string
import itertools
from Data_Processing.brandy import *
import nltk
#nltk.download('punkt')
#nltk.download('stopwords')
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import word_tokenize
tknzr = TweetTokenizer()
tknzr = TweetTokenizer(strip_handles=True, reduce_len=True)
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
stemmer = SnowballStemmer('english')
tokenizer = RegexpTokenizer(r'\w+')
stopWords = stopwords.words('english')
stopWords.append(["at-user", "at_user", "at_user", "user", "at","com", "pic", "rt", "http", "https", "url", "twitter"])
punc = string.punctuation
punc = punc.replace("'", "")
punc = punc.replace('-', '')
punc = punc.replace('_', '')
import spacy
spacy.prefer_gpu()
import scattertext as st
nlp = spacy.load("en")


#Language detection
def scoreFunction(wholetext):
    """Get text, find most common words and compare with known
    stopwords. Return dictionary of values"""
    # C makes me program like this: create always empty stuff just in case
    dictiolist = {}
    scorelist = {}
    # These are the available languages with stopwords from NLTK
    languages = ["dutch", "finnish", "german", "italian", "portuguese", "spanish", "turkish", "danish",
                   "english", "french", "hungarian", "norwegian", "russian", "swedish"]

    # Fill the dictionary of languages, to avoid  unnecessary function calls
    for lang in languages:
        dictiolist[lang] = stopwords.words(lang)

    tokens = nltk.tokenize.word_tokenize(wholetext)
    tokens = [t.lower() for t in tokens]
    freq_dist = nltk.FreqDist(tokens)

    for lang in languages:
        scorelist[lang] = 0
        for word in freq_dist.keys():
            if word in dictiolist[lang]:
                scorelist[lang] += 1
    return scorelist


def whichLanguage(scorelist):
    """This function just returns the language name, from a given
    "scorelist" dictionary as defined above."""
    maximum = 0
    try:
        for item in scorelist:
            value = scorelist[item]
            if maximum < value:
                maximum = value
                lang = item
        return lang
    except:
        return ""


def isNaN(num):
    return num != num


def find_between(s, first, last):
    try:
        start = s.rindex(first) + len(first)
        end = s.rindex(last, start)
        return s[start:end]
    except ValueError:
        return ""


def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '.', raw_html)
    return cleantext


def clean_brandwatch_chunk(data, language, sentiment, classe = True):
    # selecting  columns
    #data = data
    #selecting sentiment
    if classe == True:
        data = data.loc[data["Sentiment"] == sentiment]
        data.index = range(len(data.index))
    #selecting language
    data = data.loc[data["Language"] == language]
    data.index = range(len(data.index))
    #removing retweets
    nas = [i for i, x in enumerate(data["Full Text"]) if not str(x)=="None"]
    data = data.loc[nas]
    data.index = range(len(data.index))
    # processing reviews and counting words
    processedReviews = []
    processedSnippets = []
    wordsCount = []
    charactersCount = []
    rowsDelete = []
    foreign_languages = []

    for m in range(data.shape[0]):
        if whichLanguage(scoreFunction(data["Full Text"][int(m)])) != "english":
            foreign_languages.append(m)
            print("foreign language detected:")
            print(data["Full Text"][int(m)])
        # delting nan rows
    data.index = range(len(data.index))
    data = data.drop(foreign_languages, axis=0)
    data.index = range(len(data.index))

    for m in range(data.shape[0]):
        data["Full Text"][int(m)] = cleanhtml(data["Full Text"][int(m)])
        data["Full Text"][int(m)] = ''.join((x for x in data["Full Text"][int(m)] if x not in punc))
        line = data["Full Text"][int(m)]
        line = cleanhtml(str(line))
        # convert all urls to sting "URL"
        line = re.sub( '((www\.[^\s]+)|(https?://[^\s]+))', 'URL', line)
        line = re.sub( '@[^\s]+', 'AT_USER', line)
        line = re.sub( '[\s]+', ' ', line)
        line = re.sub( r'#([^\s]+)', r'\1', line)
        tokens = word_tokenize(line)
        reviewProcessed = ''
        for token in tokens:
            if (token.lower() not in stopWords) and (token.lower() not in ["at-user", "at_user", "at_user", "user", "at",
                             "com", "pic", "rt", "http", "https", "url", "twitter"]) and (token.lower().find(".com") == -1) \
                    and (token.lower().find("http") == -1) and not all([c.isdigit() or c == '-' for c in token]):
                token = stemmer.stem(token.lower())
                reviewProcessed += token + " "
        processedReviews.append(reviewProcessed)
        charactersCount.append(len(reviewProcessed))
        wordsCount.append(len(tokens))
    # delting nan rows
    data.index = range(len(data.index))
    data = data.drop(rowsDelete, axis=0)
    data.index = range(len(data.index))

    for m in range(data.shape[0]):
        try:
            data["Snippet"][int(m)] = cleanhtml(data["Snippet"][int(m)])
            data["Snippet"][int(m)] = ''.join((x for x in data["Snippet"][int(m)] if x not in punc))
            line = data["Snippet"][int(m)]
            line = cleanhtml(line)
            # convert all urls to sting "URL"
            line = re.sub( '((www\.[^\s]+)|(https?://[^\s]+))', 'URL', line)
            line = re.sub( '@[^\s]+', 'AT_USER', line)
            line = re.sub( '[\s]+', ' ', line)
            line = re.sub( r'#([^\s]+)', r'\1', line)
            tokens = word_tokenize(line)
            snippetProcessed = ''
            for token in tokens:
                if (token.lower() not in stopWords) and (token.lower() not in ["at-user", "at_user", "at_user", "user", "at",
                             "com", "pic", "rt", "http", "https", "url", "twitter"]) \
                        and (token.lower().find(".com") == -1) and (token.lower().find("http") == -1) and not all([c.isdigit() or c == '-' for c in token]):
                    token = stemmer.stem(token.lower())
                    snippetProcessed += token + " "
        except:
            snippetProcessed = data["Snippet"][int(m)]
        processedSnippets.append(snippetProcessed)
    data["processedReviews"] = processedReviews
    data["wordsCount"] = wordsCount
    data["charactersCount"] = charactersCount
    data["processedSnippets"] = processedSnippets
    return data


def clean_brandwatch(output_filename, input_filename, language, sentiment, add = False, classe = True):
    chunksize = 1000
    for i, chunk in enumerate(pd.read_csv(input_filename, chunksize=chunksize, error_bad_lines=False, delimiter=",", index_col=False)):
        print(i)
        data_i = clean_brandwatch_chunk(chunk, language, sentiment, classe=classe)
        if str(i) == "0" and add == False:
            with open(output_filename, 'a+', encoding="utf-8") as file:
                data_i.to_csv(file, header=True, index=False, index_label=False)
                file.close()
        else:
            with open(output_filename, 'a+', encoding="utf-8") as file:
                data_i.to_csv(file, header=False, index=False, index_label=False)
                file.close()

"""
    foreign_languages = []
    for m in range(data.shape[0]):
        if whichLanguage(scoreFunction(data["text"][int(m)])) != "english":
            print(data["text"][int(m)])
            foreign_languages.append(m)
        # delting nan rows
    data.index = range(len(data.index))
    data = data.drop(foreign_languages, axis=0)
    data.index = range(len(data.index))
"""

def clean_twitter_database(output_filename, input_filename, add=False):
    data = pd.read_csv(input_filename, error_bad_lines=False, encoding='utf-8', delimiter=",")
    data = data.drop_duplicates(subset='text', keep="last")
    data.index = range(len(data.index))

    # processing reviews and counting words
    processedReviews = []
    wordsCount = []
    charactersCount = []
    rowsDelete = []
    unselected_users = ["motor", "club", "user", "vw", "volkswagen", "news", "sale", "buy", "car",
                        "top", "leasing", "auto", ".com", "golf", "cars", "energy", "Volkswagen",
                        "car", "auto", "motor", "com", "www", "Group", "Club", "news", "vehicle",
                        "leasing", "solution", "group", "sell", "buy", "ev", "online", "network",
                        "test", "driver", "press"]

    for m in range(data.shape[0]):
        if not any([data["name"][m].lower().find(x) != -1 for x in unselected_users]):
            data["text"][m] = cleanhtml(data["text"][m])
            data["text"][m] = ''.join((x for x in data["text"][m] if x not in punc))
            line = data["text"][m]
            line = cleanhtml(line)
            # convert all urls to sting "URL"
            line = re.sub('((www\.[^\s]+)|(https?://[^\s]+))|(http?://[^\s]+)', 'URL', line)
            line = re.sub('pic[^\s]+', 'picture', line)
            line = re.sub('@[^\s]+', 'AT_USER', line)
            line = re.sub(r'#([^\s]+)', r'\1', line)
            tokens = word_tokenize(line)
            reviewProcessed = ''
            passer = False

            reviewProcessed=""
            for token in tokens:
                if token.lower() not in stopWords and passer == False and not all([c.isdigit() or c == '-' for c in token]):
                    token = stemmer.stem(token.lower())
                    reviewProcessed += token + " "
                    passer = False
            reviewProcessed = ''.join((x for x in reviewProcessed if x not in punc))
            reviewProcessed = reviewProcessed.replace("...", "")
            if len(reviewProcessed.split(" ")) > 10:
                processedReviews.append(reviewProcessed)
                charactersCount.append(len(reviewProcessed))
                wordsCount.append(len(tokens))
            else:
                rowsDelete.append(m)
        else:
            rowsDelete.append(m)
    # delting rows
    data.index = range(len(data.index))
    data = data.drop(rowsDelete, axis=0)
    data.index = range(len(data.index))
    data["processedReviews"] = processedReviews
    data["wordsCount"] = wordsCount
    data["charactersCount"] = charactersCount
    with open(output_filename, 'a+', encoding="utf-8") as file:
        data.to_csv(file, header=True, encoding="utf-8", index=False, index_label=False)
        file.close()

from mtranslate import translate

def clean_forum_database(output_filename, input_filename, add=False, language = "english"):
    data = pd.read_csv(input_filename, error_bad_lines=False, encoding='utf-8', delimiter=",")
    data.index = range(len(data.index))

    # processing reviews and counting words
    processedReviews = []
    lang = []
    rowsDelete = []

    if language != "english":
        for m in range(data.shape[0]):
            try:
                data["Text"][m] = translate(data["Text"][m], "en", language)
                print("+++++++++++++++++++++++++++++++++++++++")
                print(m)
                print("+++++++++++++++++++++++++++++++++++++++")
                print(data["Text"][m])
            except:
                print("waiting")
                time.sleep(1)

    for m in range(data.shape[0]):
        data["Date"][m] = data["Date"][m].replace("[","").replace("]","").replace('"',"")
        data["Text"][m] = cleanhtml(data["Text"][m])
        data["Text"][m] = ''.join((x for x in data["Text"][m] if x not in punc))
        line = data["Text"][m].replace(".", ";").replace(",", ";").replace("r-link", "rlink").replace("r_link", "rlink").replace("r link", "rlink")
        # convert all urls to sting "URL"
        line = re.sub('((www\.[^\s]+)|(https?://[^\s]+))|(http?://[^\s]+)', 'URL', line)
        line = re.sub('pic[^\s]+', 'picture', line)
        line = re.sub('@[^\s]+', 'AT_USER', line)
        line = re.sub(r'#([^\s]+)', r'\1', line)
        tokens = word_tokenize(line)
        reviewProcessed = ''
        passer = False

        reviewProcessed=""
        for token in tokens:
            if token.lower() not in stopWords and passer == False and not all([c.isdigit() or c == '-' for c in token]):
                token = stemmer.stem(token.lower())
                reviewProcessed += token + " "
                passer = False
        reviewProcessed = ''.join((x for x in reviewProcessed if x not in punc))
        reviewProcessed = reviewProcessed.replace("...", "")
        if len(reviewProcessed.split(" ")) > 10:
            processedReviews.append(reviewProcessed)
            lang.append(language)
        else:
            rowsDelete.append(m)
    # delting rows
    data.index = range(len(data.index))
    data = data.drop(rowsDelete, axis=0)
    data.index = range(len(data.index))
    data["processedReviews"] = processedReviews
    data["language"] = lang

    with open(output_filename, 'a+', encoding="utf-8") as file:
        data.to_csv(file, header=True, encoding="utf-8", index=False, index_label=False)
        file.close()



def clean_forum_database_chunck(output_filename, input_filename, add=False, language = "english"):
    data_chunk = pd.read_csv(input_filename, chunksize= 10000, error_bad_lines=False, encoding='utf-8', delimiter=",")
    for i, data in enumerate(data_chunk):
        data.index = range(len(data.index))

        # processing reviews and counting words
        processedReviews = []
        lang = []
        rowsDelete = []

        if language != "english":
            for m in range(data.shape[0]):
                try:
                    data["Text"][m] = translate(data["Text"][m], "en", language)
                    print("+++++++++++++++++++++++++++++++++++++++")
                    print(m)
                    print("+++++++++++++++++++++++++++++++++++++++")
                    print(data["Text"][m])
                except:
                    print("waiting")
                    time.sleep(1)

        for m in range(data.shape[0]):
            data["Date"][m] = data["Date"][m].replace("[","").replace("]","").replace('"',"")
            data["Text"][m] = cleanhtml(data["Text"][m])
            data["Text"][m] = ''.join((x for x in data["Text"][m] if x not in punc))
            line = data["Text"][m].replace(".", ";").replace(",", ";").replace("r-link", "rlink").replace("r_link", "rlink").replace("r link", "rlink")
            # convert all urls to sting "URL"
            line = re.sub('((www\.[^\s]+)|(https?://[^\s]+))|(http?://[^\s]+)', 'URL', line)
            line = re.sub('pic[^\s]+', 'picture', line)
            line = re.sub('@[^\s]+', 'AT_USER', line)
            line = re.sub(r'#([^\s]+)', r'\1', line)
            tokens = word_tokenize(line)
            reviewProcessed = ''
            passer = False

            reviewProcessed=""
            for token in tokens:
                if token.lower() not in stopWords and passer == False and not all([c.isdigit() or c == '-' for c in token]):
                    token = stemmer.stem(token.lower())
                    reviewProcessed += token + " "
                    passer = False
            reviewProcessed = ''.join((x for x in reviewProcessed if x not in punc))
            reviewProcessed = reviewProcessed.replace("...", "")
            if len(reviewProcessed.split(" ")) > 10:
                processedReviews.append(reviewProcessed)
                lang.append(language)
            else:
                rowsDelete.append(m)
        # delting rows
        data.index = range(len(data.index))
        data = data.drop(rowsDelete, axis=0)
        data.index = range(len(data.index))
        data["processedReviews"] = processedReviews
        data["language"] = lang

        with open(output_filename + str(i) + ".csv", 'a+', encoding="utf-8") as file:
            data.to_csv(file, header=True, encoding="utf-8", index=False, index_label=False)
            file.close()
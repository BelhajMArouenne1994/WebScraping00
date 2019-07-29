from mtranslate import translate
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
stopWords.append(["at-user", "at_user", "com", "http", "https", "twitter"])
punc_processed = string.punctuation
punc_processed = punc_processed.replace("'", "")
punc_processed = punc_processed.replace('-', '')
punc_processed = punc_processed.replace('_', '')

punc = string.punctuation
punc = punc.replace("'", "")
punc = punc.replace('-', '')
punc = punc.replace('_', '')
punc = punc.replace(',', '')
punc = punc.replace('.', '')
punc = punc.replace(';', '')
punc = punc.replace('!', '')
punc = punc.replace('?', '')

import spacy
spacy.prefer_gpu()
import scattertext as st
nlp = spacy.load("en")

from googletrans import Translator

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



def clean_posts_database(output_filename, input_filename, add=False, language = "english"):
    data = pd.read_csv(input_filename, error_bad_lines=False, encoding='utf-8', delimiter=",")
    data.index = range(len(data.index))

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

    # processing reviews and counting words
    processedReviews = []
    wordsCount = []
    charactersCount = []
    rowsDelete = []
    lang = []

    for m in range(data.shape[0]):
        data["Date"][m] = str(data["Date"][m]).replace("[", "").replace("]", "").replace('"', "")
        if data["Date"][m][0:3] != "201":
            data["Date"][m] = "2019-06-20"
        data["Text"][m] = cleanhtml(data["Text"][m])
        data["Text"][m] = ''.join((x for x in data["Text"][m] if x not in punc))
        data["Text"][m] = data["Text"][m].lower().replace(".", ";").replace(",", ";").replace("r-link", "rlink").replace("r_link", "rlink").replace("r link", "rlink")
        line = data["Text"][m].lower()
        # convert all urls to sting "URL"
        line = re.sub('((www\.[^\s]+)|(https?://[^\s]+))|(http?://[^\s]+)', 'URL', line)
        tokens = word_tokenize(line)
        reviewProcessed = ''
        passer = False

        reviewProcessed=""
        for token in tokens:
            if token.lower() not in stopWords and passer == False and not all([c.isdigit() or c == '-' for c in token]):
                token = stemmer.stem(token.lower())
                reviewProcessed += token + " "
                passer = False
        reviewProcessed = ''.join((x for x in reviewProcessed if x not in punc_processed))
        reviewProcessed = reviewProcessed.replace("...", "")
        if len(reviewProcessed.split(" ")) > 10:
            lang.append(language)
            processedReviews.append(reviewProcessed)
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
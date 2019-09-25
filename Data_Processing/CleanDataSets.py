import os
import re
import string

import pandas as pd
import spacy
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize

stemmer = SnowballStemmer('english')
tokenizer = RegexpTokenizer(r'\w+')
stopWords = stopwords.words('english')
stopWords.append(["at-user", "at_user", "at_user", "user", "at", "com", "pic", "rt", "http", "https", "url", "twitter"])
punc = string.punctuation
punc = punc.replace("'", "")
punc = punc.replace('-', '')
punc = punc.replace('_', '')
nlp = spacy.load('en_core_web_sm')


def isNaN(num):
    return num != num


def find_between(s, first, last):
    try:
        start = s.rindex(first)+len(first)
        end = s.rindex(last, start)
        return s[start:end]
    except ValueError:
        return ""


def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '.', raw_html)
    cleantext = cleantext.replace("\n", " ").replace("\r", " ")
    return cleantext


def clean_dataset(output_filename, input_filename):
    try:
        os.remove(r"Data_Processing\Outputs\DataSets\Dataset_clean.csv")
    except:
        pass
    """
    directory = r"Data_Processing\Outputs\."
    try:
        os.mkdir(directory)
    except FileExistsError:
        pass

    directory = r"Data_Processing\Outputs\DataSets\."
    try:
        # Create target Directory
        os.mkdir(directory)
    except FileExistsError:
        pass
    """

    data = pd.read_csv(input_filename, error_bad_lines=False, encoding='utf-8', delimiter="µ", engine='python')
    data.index = range(len(data.index))

    # processing reviews and counting words
    processedReviews = []
    rowsDelete = []

    for m in range(data.shape[0]):
        try:
            data["Text"][m] = cleanhtml(data["Text"][m])
        except:
            data["Text"][m] = ""
        line = ''.join((x for x in data["Text"][m] if x not in punc))

        # convert all urls to sting "URL"
        line = re.sub('((www\.[^\s]+)|(https?://[^\s]+))|(http?://[^\s]+)', 'URL', line)
        line = re.sub('pic[^\s]+', 'picture', line)
        line = re.sub('@[^\s]+', 'AT_USER', line)
        line = re.sub(r'#([^\s]+)', r'\1', line)
        line = ''.join((x for x in line if x not in ['URL', 'picture', 'AT_USER', r'\1']))

        tokens = word_tokenize(line)

        reviewProcessed = ""
        for token in tokens:
            if token.lower() not in stopWords and token.lower() not in punc and not all(
                    [c.isdigit() or c == '-' for c in token]):
                token = stemmer.stem(token.lower())
                reviewProcessed += token+" "

        reviewProcessed = reviewProcessed.replace("...", "").replace("\n", " ").replace("\r", " ")

        if len(reviewProcessed.split(" ")) > 10:
            processedReviews.append(reviewProcessed)
        else:
            rowsDelete.append(m)

    # delting rows
    data.index = range(len(data.index))
    data = data.drop(rowsDelete, axis=0)
    data.index = range(len(data.index))
    data["processedReviews"] = processedReviews

    with open(output_filename+".csv", 'a+', encoding="utf-8") as file:
        data.to_csv(file, header=True, encoding="utf-8", index=False, index_label=False)
        file.close()

    message = ""
    try:
        data = pd.read_csv(output_filename+".csv", error_bad_lines=False, encoding='utf-8', delimiter=",")
        if data.shape[0] > 0:
            message = "Processus de nettoyage fini avec succés."
        elif data.shape[0] == 0:
            message = "Erreur lors du processus de Nettoyage de la base."
    except:
        message = "Erreur lors du processus de Nettoyage de la base."

    return message


def clean_dataset_chunck(output_filename, input_filename):
    try:
        os.remove(r"Data_Processing\Outputs\DataSets\Dataset_clean.csv")
    except:
        pass

    data_chunk = pd.read_csv(input_filename, chunksize=10000, error_bad_lines=False, encoding='utf-8', delimiter="µ",
                             engine='python')
    for i, data in enumerate(data_chunk):
        data.index = range(len(data.index))
        # processing reviews and counting words
        processedReviews = []
        rowsDelete = []

        for m in range(data.shape[0]):
            try:
                line = cleanhtml(data["Text"][m])
                data["Text"][m] = line
                line = ''.join((x for x in data["Text"][m] if x not in punc))

                # convert all urls to sting "URL"
                line = re.sub('((www\.[^\s]+)|(https?://[^\s]+))|(http?://[^\s]+)', 'URL', line)
                line = re.sub('pic[^\s]+', 'picture', line)
                line = re.sub('@[^\s]+', 'AT_USER', line)
                line = re.sub(r'#([^\s]+)', r'\1', line)
                line = ''.join((x for x in line if x not in ['URL', 'picture', 'AT_USER', r'\1']))

                tokens = word_tokenize(line)

                reviewProcessed = ""
                for token in tokens:
                    if token.lower() not in stopWords and token.lower() not in punc and not all(
                            [c.isdigit() or c == '-' for c in token]):
                        token = stemmer.stem(token.lower())
                        reviewProcessed += token+" "
            except:
                pass

            reviewProcessed = reviewProcessed.replace("...", "").replace("\n", " ").replace("\r", " ")

            if len(reviewProcessed.split(" ")) > 10:
                processedReviews.append(str(reviewProcessed))
            else:
                rowsDelete.append(m)

        data.index = range(len(data.index))
        data = data.drop(rowsDelete, axis=0)
        data.index = range(len(data.index))
        data["processedReviews"] = processedReviews

        header = False
        if i == 0:
            header = True

        with open(output_filename+".csv", 'a+', encoding="utf-8") as file:
            data.to_csv(file, header=header, encoding="utf-8", index=False, index_label=False)
            file.close()

    message = ""
    try:
        data = pd.read_csv(output_filename+".csv", error_bad_lines=False, encoding='utf-8', delimiter=",")
        if data.shape[0] > 0:
            message = "Processus de nettoyage fini avec succés."
        elif data.shape[0] == 0:
            message = "Erreur lors du processus de Nettoyage de la base."
    except:
        message = "Erreur lors du processus de Nettoyage de la base."

    return message

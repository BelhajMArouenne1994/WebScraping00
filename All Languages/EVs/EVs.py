from Data_Processing.CleanDataSets import *
from Data_Processing.brandy import *
from Data_Processing.reviews_segmentation import *
import scattertext as st
import spacy
import matplotlib.pyplot as plot
import numpy as np
import pandas as pd
import tensorflow as tf
import multiprocessing
import keras.backend as K
import datetime
import gensim
from gensim.models import Word2Vec
import pprint

# Select whether using Keras with or without GPU support
use_gpu = True
config = tf.ConfigProto(intra_op_parallelism_threads=multiprocessing.cpu_count(),
                        inter_op_parallelism_threads=multiprocessing.cpu_count(),
                        allow_soft_placement=True,
                        device_count={'CPU': 1,
                                      'GPU': 1 if use_gpu else 0})
session = tf.Session(config=config)
K.set_session(session)
os.chdir(r"/home/marwen/PycharmProjects/Renault/ZOE/EVs")
###-------------------------------------------------###


###--- Downloading mentions from Brandwatch API ---###
user_name = "marouenne.belhaj@renault.com"
password = "marwen1A?"
access_token = get_new_key(user_name, password)
project_list = boot_brandy(access_token)

project_name = "EVs"
project_id = get_project_id_from_name(project_list, project_name)
query_list = get_query_id(project_id, access_token)
query_id = "1999778052"
stream_data( "/home/marwen/PycharmProjects/Renault/ZOE/EVs/EVs_brandWatch.csv", '2012-02-02', str(datetime.datetime.today().date()), project_id, query_id, access_token, new=True)
query_id = "1999777723"
stream_data( "/home/marwen/PycharmProjects/Renault/ZOE/EVs/EVs_brandWatch.csv", '2012-02-02', str(datetime.datetime.today().date()), project_id, query_id, access_token, new=False)
query_id = "1999772246"
stream_data( "/home/marwen/PycharmProjects/Renault/ZOE/EVs/EVs_brandWatch.csv", '2012-02-02', str(datetime.datetime.today().date()), project_id, query_id, access_token, new=False)

project_name = "HEV"
project_id = get_project_id_from_name(project_list, project_name)
query_list = get_query_id(project_id, access_token)
query_id = "1998266389"
stream_data( "/home/marwen/PycharmProjects/Renault/ZOE/EVs/HEVs_brandWatch.csv", '2012-02-02', str(datetime.datetime.today().date()), project_id, query_id, access_token, new=True)

project_name = "PHEV"
project_id = get_project_id_from_name(project_list, project_name)
query_list = get_query_id(project_id, access_token)
query_id = "1998266388"
stream_data("/home/marwen/PycharmProjects/Renault/ZOE/EVs/PHEVs_brandWatch.csv", '2012-02-02', str(datetime.datetime.today().date()), project_id, query_id, access_token, new=True)
###-------------------------------------------------###


###-------------------------------------------------###
clean_brandwatch("EVs_clean.csv", "EVs_brandWatch.csv", "en", "", add=False, classe=False)
clean_brandwatch("EVs_clean.csv", "HEVs_brandWatch.csv", "en", "", add=True, classe=False)
clean_brandwatch("EVs_clean.csv", "PHEVs_brandWatch.csv", "en", "", add=True, classe=False)

BrandWatch_texts = pd.read_csv("EVs/EVs_clean.csv", error_bad_lines=False, encoding='utf-8', delimiter=",", index_col=False)

with open("EVs/EVs_Reviews.csv", 'a+') as f:
    f.write("ReviewsµprocessedReviews")
    f.write("\n")
    f.close()
for i in range(BrandWatch_texts.shape[0]):
    if int(BrandWatch_texts["wordsCount"][i]) > 100:
        if BrandWatch_texts["processedSnippets"][i] == BrandWatch_texts["processedSnippets"][i]:
            with open("EVs/EVs_Reviews.csv", 'a+') as f:
                f.write(BrandWatch_texts["Snippet"][i].replace("\n", " ").replace('"', " ").replace('"', " ").replace("'", " ").replace(",", "."))
                f.write("µ")
                f.write(BrandWatch_texts["processedSnippets"][i] + " ")
                f.write("\n")
            f.close()
    else:
        if BrandWatch_texts["processedReviews"][i] == BrandWatch_texts["processedReviews"][i]:
            with open("EVs/EVs_Reviews.csv", 'a+') as f:
                f.write(BrandWatch_texts["Full Text"][i].replace("\n", " ").replace('"', " ").replace("'", " ").replace(",", "."))
                f.write("µ")
                f.write(BrandWatch_texts["processedReviews"][i])
                f.write("\n")
            f.close()
del BrandWatch_texts
###---------------------------------------------------------------------------###


###----------------------------Removing Duplicates----------------------------###
processedReviews = pd.read_csv("EVs/EVs_Reviews.csv", delimiter="µ")

# Python code to remove duplicate elements
def Remove(duplicate):
    final_list = []
    indexes = []
    for i, num in enumerate(duplicate['processedReviews']):
        if num not in final_list:
            final_list.append(num)
            indexes.append(i)
    return duplicate.loc[indexes]

processedReviews_clean = Remove(processedReviews)
with open("EVs/EVs_Reviews_2.csv", 'w+', encoding="utf-8") as file:
    processedReviews_clean.to_csv(file, header=True, encoding="utf-8", index=False, index_label=False)
    file.close()
del processedReviews
del processedReviews_clean
###-------------------------------###


"""
###---------------------------------- Word2Vec Model Construction----------------------------------###
key_words = ["ev", "vehicle", "drive",  "brake", "battery", "plug-in", "hybrid", "electric",
             "HEV", "PHEV", "Fuel", "charging", "connector", "home", "phase"]

class corpus(object):
    def __iter__(self):
        for line in pd.read_csv('EVs/EVs_Reviews_2.csv', delimiter=',', error_bad_lines=False, encoding="utf-8")["processedReviews"]:
            yield line.lower().split()

model = gensim.models.Word2Vec(corpus(), size=256, window=5, min_count=100, workers=multiprocessing.cpu_count(), iter=50)
model.save("EVs_w2v.model")
"""
"""

model = Word2Vec.load("/home/marwen/PycharmProjects/Renault/ZOE/EVs_word2vec(128).model")
print("Positive")
similar_words = {search_term: [item[0] for item in model.wv.most_similar(positive=[search_term], topn=20)]
                 for search_term in [stemmer.stem("autonomy")]}
pprint.pprint(similar_words, compact=True)

print("Positive")
similar_words = {search_term: [item[0] for item in model.wv.most_similar(positive=[search_term], topn=20)]
                 for search_term in [stemmer.stem("gearbox")]}
pprint.pprint(similar_words, compact=True)

print("Positive")
similar_words = {search_term: [item[0] for item in model.wv.most_similar(positive=[search_term], topn=10)]
                 for search_term in [stemmer.stem("door")]}
pprint.pprint(similar_words, compact=True)


print("Positive")
similar_words = {search_term: [item[0] for item in model.wv.most_similar(positive=[search_term], topn=120)]
                 for search_term in [stemmer.stem("passeng")]}
pprint.pprint(similar_words, compact=True)


for key_word in key_words:
    try:
        print("Positive")
        similar_words = {search_term: [item[0] for item in model.wv.most_similar(positive=[search_term], topn=10)]
                         for search_term in [stemmer.stem(key_word)]}
        pprint.pprint(similar_words, compact=True)
    except:
        pass
"""
###-------------------------------------------------------------------------------------------------###
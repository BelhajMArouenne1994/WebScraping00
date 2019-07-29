from Data_Processing.CleanDataSets import *
from Data_Processing.brandy import *
from Data_Processing.reviews_segmentation_seats import *
import scattertext as st
import spacy
import matplotlib.pyplot as plot
import numpy as np
import pandas as pd
import multiprocessing
import datetime
import gensim
from gensim.models import Word2Vec
import pprint
from nltk.sentiment.vader import SentimentIntensityAnalyzer
#nltk.download('vader_lexicon')

os.chdir(r"/home/marwen/PycharmProjects/Renault/CARS/ZOE/ZOE (Forums) (anglais)")
###-------------------------------------------------###


#categories = ["battery", "brakes", "dealer", "doors-interior-seats", "engine-gearbox", "equipements", "price"]
#[item[0] for item in model.wv.most_similar(positive=[stemmer.stem("cap")], topn=100)]
categories = ["seats"]
seats = ["seat", "seats", "passag"]
#flap = ["flap", "cap", "plastic", "port", "cover", "lid", "reservoir", "inlet"]
model = Word2Vec.load("Word2Vec/models/EVs_word2vec (10 iters).model")

###-------------------------------------------------###
data = pd.read_csv("intermediate/Reviews_final.csv", error_bad_lines=False)
list_stop_words = ["zoe", "renault", "car", "vehicle", "vw", "electric-car"]

preds = []
for i in range(data.shape[0]):
    words = str(data["processedReviews"][i]).split()
    counts = []
    results_seats = 0
    results_flap = 0
    for w in words:
        if w not in list_stop_words:
            results_seats += seats.count(w)
            #results_flap += flap.count(w)

    results_seats = results_seats / len(seats)
    #results_flap = results_flap / len(flap)

    #counts = [results_seats, results_flap]
    counts = [results_seats]

    if max(counts) > 0:
        categorie = categories[np.argmax(counts)]
        preds.append(categorie)
    else:
        preds.append(None)

data_custered = pd.DataFrame(list(zip(data["Reviews"], data["processedReviews"],
                                      data["Date"], data["Sentiments"], data["Lang"], preds)),
                             columns=['Reviews', 'processedReviews', 'Date', 'Sentiments', 'Lang', 'Cluster'])

data_custered.to_csv("ZOE_seats.csv", header=True, index=False, index_label=False)

########################################################################################################################

###------------------------------------------------------------------------------------------------------------------###
sent = pd.read_csv("ZOE_seats.csv", error_bad_lines=False)
sent['Sentiments'] = sent['Sentiments'].astype(str)
negatives = sent.loc[sent["Sentiments"] == "negative"]
negatives.index = range(len(negatives.index))
with open("ZOE_seats_negatives.csv", 'w+', encoding="utf-8") as file:
    negatives.to_csv(file, header=True, index=False, index_label=False)
    file.close()
###------------------------------------------------------------------------------------------------------------------###

Plot_Clusters_trained("ZOE_seats.csv", "")
Plot_Time_Series("ZOE_seats.csv")
wordcloud_cluster_byIds("ZOE_seats.csv")
#############################################


###--- Clustering Data ---###
"""
    Clustering Data and Deleting those which are not related to our Topic
"""
directory = "Kmeans"
try:
    # Create target Directory
    os.mkdir(directory)
    print("Directory ", directory, " Created ")
except FileExistsError:
    print("Directory ", directory, " already exists")


def flatten(directory, processedReviews_file, max_df):
    try:
        # Create target Directory
        os.mkdir(directory)
        print("Directory ", directory, " Created ")
    except FileExistsError:
        print("Directory ", directory, " already exists")
    text2kw_keywords = extract_tfidf_keywords(directory + "/keywords.csv", processedReviews_file, 20, max_df)
    text2kw_coefs = extract_tfidf_coefs(directory + "/coefs.csv", processedReviews_file, 20, max_df)
    top_n_keywords_average(directory + "/text2kw_top_n_keywords.csv", model, directory + "/keywords.csv",
                           directory + "/coefs.csv",
                           vector_size=128)
    ElbowMethod(directory + "/", directory + "/text2kw_top_n_keywords.csv", 15)
###-----------------------###

for categorie in categories:
    path = "Kmeans/" + categorie
    cat = pd.read_csv("ZOE_clustered.csv", error_bad_lines=False)
    cat['Cluster'] = cat['Cluster'].astype(str)
    cat = cat.loc[sent["Cluster"] == str(categorie)]
    cat.index = range(len(cat.index))
    with open(str(categorie) + ".csv", 'w+', encoding="utf-8") as file:
        cat.to_csv(file, header=True, index=False, index_label=False)
        file.close()
    flatten(path, str(categorie) + ".csv", max_df=0.8)

    for i in [4, 5, 6, 7]:
        path_plots = "Kmeans/" + categorie + "/Clusters_" + str(categorie) + "_" + str(i)
        try:
            # Create target Directory
            os.mkdir(path_plots)
            print("Directory ", path_plots, " Created ")
        except FileExistsError:
            print("Directory ", path_plots, " already exists")

        Plot_Clusters_Kmeans(path_plots + "/Reviews_Clusters.csv", str(categorie) + ".csv",
                             path + "/text2kw_top_n_keywords.csv", i, 1, path_plots + "/")

        wordcloud_cluster_byIds_Kmeans(path_plots, path_plots + "/Reviews_Clusters.csv", i)
        Plot_Time_Series_Kmeans(path_plots, i)

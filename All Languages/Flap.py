from Data_Processing.CleanDataSets import *
from Data_Processing.brandy import *
from Data_Processing.reviews_segmentation_flap import *
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
categories = ["flap"]
flap = ["flap", "flaps", "hatch", "lid", "inlet", "socket", "port", "lock", "unlock", "disconnect", "open",
        "close", "bonnet", "locked", "blocked"]
others = ["app", "application", "door", "doors", "window", "windows", "msg", "message", "rlink"]
model = Word2Vec.load("Word2Vec/models/EVs_word2vec (10 iters).model")

###-------------------------------------------------###
data = pd.read_csv("intermediate/Reviews_final.csv", error_bad_lines=False)
list_stop_words = ["zoe", "renault", "car", "vehicle", "vw", "electric-car"]

preds = []
for i in range(data.shape[0]):
    words = str(data["processedReviews"][i]).split()
    counts = []
    results_flap = 0
    for w in words:
        if w not in list_stop_words:
            results_flap += flap.count(w)
    results_flap = results_flap / len(flap)
    counts = [results_flap]

    words = str(data["processedReviews"][i]).split()
    results_others = 0
    for w in words:
        if w not in list_stop_words:
            results_others += others.count(w)
    results_others = results_others / len(others)

    if max(counts) > 0 and results_others == 0:
        categorie = categories[np.argmax(counts)]
        preds.append(categorie)
    else:
        preds.append(None)

data_custered = pd.DataFrame(list(zip(data["Reviews"], data["processedReviews"],
                                      data["Date"], data["Sentiments"], data["Lang"], preds)),
                             columns=['Reviews', 'processedReviews', 'Date', 'Sentiments', 'Lang', 'Cluster'])

data_custered.to_csv("ZOE_flap.csv", header=True, index=False, index_label=False)

########################################################################################################################

###------------------------------------------------------------------------------------------------------------------###
sent = pd.read_csv("ZOE_flap.csv", error_bad_lines=False)
sent['Cluster'] = sent['Cluster'].astype(str)
negatives = sent.loc[sent["Cluster"] == "flap"]
negatives.index = range(len(negatives.index))
with open("ZOE_flap_only.csv", 'w+', encoding="utf-8") as file:
    negatives.to_csv(file, header=True, index=False, index_label=False)
    file.close()
###------------------------------------------------------------------------------------------------------------------###

###------------------------------------------------------------------------------------------------------------------###
sent = pd.read_csv("ZOE_flap_only.csv", error_bad_lines=False)
sent['Sentiments'] = sent['Sentiments'].astype(str)
negatives = sent.loc[sent["Sentiments"] == "negative"]
negatives['Cluster'] = sent['Cluster'].astype(str)
negatives = negatives.loc[sent["Cluster"] == "flap"]
negatives.index = range(len(negatives.index))
with open("ZOE_flap_negatives.csv", 'w+', encoding="utf-8") as file:
    negatives.to_csv(file, header=True, index=False, index_label=False)
    file.close()
###------------------------------------------------------------------------------------------------------------------###

Plot_Clusters_trained("ZOE_flap_only.csv", "")
Plot_Time_Series("ZOE_flap_negatives.csv")
wordcloud_cluster_byIds("ZOE_flap_only.csv")

all = pd.read_csv("ZOE_flap_only_final.csv", encoding='utf8', delimiter=";")

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

    for i in [4]:
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

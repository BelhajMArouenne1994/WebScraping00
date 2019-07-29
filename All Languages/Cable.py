from Data_Processing.brandy import *
from Data_Processing.reviews_segmentation_seats import *
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
os.chdir(r"/home/marwen/PycharmProjects/Renault/CARS/ZOE/ZOE (Forums) (anglais)")
###-------------------------------------------------###


#categories = ["battery", "brakes", "dealer", "doors-interior-seats", "engine-gearbox", "equipements", "price"]
#[item[0] for item in model.wv.most_similar(positive=[stemmer.stem("cap")], topn=100)]
categories = ["cable"]
cable = [stemmer.stem("cable"), stemmer.stem('connector'), stemmer.stem('socket'), stemmer.stem("outlet")]
insert = [stemmer.stem("insert"), stemmer.stem('put'), stemmer.stem('plug'), stemmer.stem("unplug"),
          stemmer.stem("block"), stemmer.stem("blocked")]

model = Word2Vec.load("Word2Vec/models/EVs_word2vec (10 iters).model")

###-------------------------------------------------###
data = pd.read_csv("all.csv", error_bad_lines=False)
list_stop_words = ["zoe", "renault", "car", "vehicle", "vw", "electric-car"]

preds = []
for i in range(data.shape[0]):
    words = str(data["processedReviews"][i]).split()
    counts = []
    results_seats = 0
    results_insert = 0
    for w in words:
        if w not in list_stop_words:
            results_seats += cable.count(w)
            results_insert += insert.count(w)

    results_seats = results_seats / len(cable)
    results_insert = results_insert / len(insert)

    #counts = [results_seats, results_flap]
    counts_seats = [results_seats]
    counts_insert = [results_insert]

    if max(counts_seats) > 0 and max(counts_insert) > 0:
        categorie = categories[np.argmax(counts_seats)]
        preds.append(categorie)
    else:
        preds.append(None)

data_custered = pd.DataFrame(list(zip(data["Reviews"], data["processedReviews"],
                                      data["Date"], data["Sentiments"], data["Lang"], preds)),
                             columns=['Reviews', 'processedReviews', 'Date', 'Sentiments', 'Lang', 'Cluster'])

data_custered.to_csv("ZOE_cable.csv", header=True, index=False, index_label=False)

########################################################################################################################

###------------------------------------------------------------------------------------------------------------------###
sent = pd.read_csv("ZOE_cable.csv", error_bad_lines=False)
sent['Cluster'] = sent['Cluster'].astype(str)
negatives = sent.loc[sent["Cluster"] == "cable"]
with open("ZOE_cable_only.csv", 'w+', encoding="utf-8") as file:
    negatives.to_csv(file, header=True, index=False, index_label=False)
    file.close()
###------------------------------------------------------------------------------------------------------------------###

Plot_Clusters_trained("ZOE_cable_only.csv", "")
Plot_Time_Series("ZOE_seats.csv")
wordcloud_cluster_byIds("ZOE_seats.csv")
#############################################


###--- Clustering Data ---###
"""
    Clustering Data and Deleting those which are not related to our Topic
"""
directory = "Kmeans_cable"
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
    path = "Kmeans_cable/" + categorie
    cat = pd.read_csv("ZOE_cable_only.csv", error_bad_lines=False)
    cat['Cluster'] = cat['Cluster'].astype(str)
    cat = cat.loc[sent["Cluster"] == str(categorie)]
    cat.index = range(len(cat.index))
    with open(str(categorie) + ".csv", 'w+', encoding="utf-8") as file:
        cat.to_csv(file, header=True, index=False, index_label=False)
        file.close()
    flatten(path, "ZOE_cable_only.csv", max_df=0.8)

    for i in [2, 3]:
        path_plots = "Kmeans_cable/" + categorie + "/Clusters_" + str(categorie) + "_" + str(i)
        try:
            # Create target Directory
            os.mkdir(path_plots)
            print("Directory ", path_plots, " Created ")
        except FileExistsError:
            print("Directory ", path_plots, " already exists")

        Plot_Clusters_Kmeans(path_plots + "/Reviews_Clusters.csv",   "ZOE_cable_only.csv",
                             path + "/text2kw_top_n_keywords.csv", i, 1, path_plots + "/")

        wordcloud_cluster_byIds_Kmeans(path_plots, path_plots + "/Reviews_Clusters.csv", i)
        #Plot_Time_Series_Kmeans(path_plots, i)


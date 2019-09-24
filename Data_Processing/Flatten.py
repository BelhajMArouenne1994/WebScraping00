from Data_Processing.ReviewsSegmentation import *
import pandas as pd
from gensim.models import Word2Vec
import os
from os import listdir
from os.path import isfile, join


def Reviews_Flatten(DATA_PATH, path):

    Data = pd.read_csv(DATA_PATH, error_bad_lines=False, encoding='utf-8', delimiter=",", engine="python")

    diff_score = score_Review(Data["Text"])
    sentiments = pd.concat([Data, diff_score], axis=1)
    sentiments.columns = ['Car', 'Country', 'Text', 'Date', 'processedReviews', "compound", "neg", "neu", "pos",
                          "Sentiments"]

    sentiments.index = range(len(sentiments.index))

    del Data
    del diff_score

    try:
        os.remove(path + r"\Outputs\DataSets\Reviews_final.csv")
    except:
        pass

    with open(path + r"\Outputs\DataSets\Reviews_final.csv", 'w+',
              encoding="utf-8") as file:
        sentiments.to_csv(file, header=True, encoding="utf-8", index=False, index_label=False)
        file.close()


    ###--- Clustering Data ---###
    """
        Clustering Data and Deleting those which are not related to our Topic
    """

    def flatten(processedReviews_file, max_df):

        directory = path + r"\Outputs\Flatten"
        try:
            # Create target Directory
            os.mkdir(directory)
        except FileExistsError:
            pass

        extract_tfidf_keywords(directory + r"\keywords.csv", processedReviews_file, 20, max_df)
        extract_tfidf_coefs(directory + r"\coefs.csv", processedReviews_file, 20, max_df)
        model = Word2Vec.load("Word2vVec.model")
        top_n_keywords_average(directory + r"\text2kw_top_n_keywords.csv", model, directory + r"\keywords.csv",
                               directory + r"\coefs.csv", vector_size=256)
        ElbowMethod(directory, directory + r"\text2kw_top_n_keywords.csv", 15)

        results = ['coefs.csv', 'ElbowCurve.png', 'keywords.csv', 'text2kw_top_n_keywords.csv']
        onlyfiles = [f for f in listdir(directory) if isfile(join(directory, f))]
        intersection = set(results) & set(onlyfiles)
        if len(intersection) == len(results):
            return "Data Flattened with success"
        else:
            return "error"


    return flatten(path + r"\Outputs\DataSets\Reviews_final.csv", max_df=0.9)
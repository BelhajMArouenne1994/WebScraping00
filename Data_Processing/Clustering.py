import pandas as pd
import numpy as np
import csv
from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer('english')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
# nltk.download('vader_lexicon')
from nltk.cluster import KMeansClusterer
import spacy

spacy.prefer_gpu()
nlp = spacy.load('en_core_web_sm')
from numpy import genfromtxt
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
import collections
from nltk.corpus import stopwords

stopWordsFrench = list(set(stopwords.words('french')))
stopWordsGerman = list(set(stopwords.words('german')))
ENGLISH_STOP_WORDS = list(ENGLISH_STOP_WORDS)
ENGLISH_STOP_WORDS.append("hello")
ENGLISH_STOP_WORDS.append("evening")
ENGLISH_STOP_WORDS.append("thank")
ENGLISH_STOP_WORDS.append("thanks")
ENGLISH_STOP_WORDS.append("love")
ENGLISH_STOP_WORDS.append("club")
ENGLISH_STOP_WORDS.append("hi")
ENGLISH_STOP_WORDS.append("forum")
ENGLISH_STOP_WORDS = list(tuple(ENGLISH_STOP_WORDS))
stopWords = stopwords.words('english')
stopWords = stopWords + ENGLISH_STOP_WORDS + stopWordsGerman + stopWordsFrench

import scattertext as st
from nltk.sentiment.vader import SentimentIntensityAnalyzer
# nltk.download('vader_lexicon')
from nltk.tokenize import word_tokenize

import itertools
from gensim.models import Word2Vec

model = Word2Vec.load("Word2vVec.model")

import spacy
from scattertext import word_similarity_explorer_gensim

nlp = spacy.load('en_core_web_sm')

import os
from os import listdir
from os.path import isfile, join


def isNaN(num):
    return num != num


def top_tfidf_feats(row, features, top_n=25):
    topn_ids = np.argsort(row)[::-1][:top_n]
    top_feats = [(features[i], row[i]) for i in topn_ids]
    df = pd.DataFrame(top_feats)
    df.columns = ['feature', 'tfidf']
    return df


def extract_tfidf_keywords(output_filename, input_filename, top_n, max_df=1.0, min_df=0.0):
    df = pd.read_csv(input_filename, sep=',')
    try:
        os.remove(output_filename)
    except:
        pass

    with open(output_filename, 'a+', encoding="utf-8") as f:
        tfidf_vectorizer = TfidfVectorizer(max_df=max_df, max_features=2000000,
                                           min_df=min_df, stop_words="english",
                                           use_idf=True, ngram_range=(1, 1))
        tfidf_matrix = tfidf_vectorizer.fit_transform(df['processedReviews'].values.astype('U'))
        terms = tfidf_vectorizer.get_feature_names()
        for i in range(0, tfidf_matrix.shape[0]):
            row = np.squeeze(tfidf_matrix[i].toarray())
            feats = top_tfidf_feats(row, terms, top_n)
            f.write(' '.join(str(e) for e in feats["feature"]))
            f.write("\n")


def extract_tfidf_coefs(output_filename, input_filename, top_n, max_df=1.0, min_df=0.0):
    df = pd.read_csv(input_filename, sep=',', chunksize=10000)
    try:
        os.remove(output_filename)
    except:
        pass
    for df_i in df:
        with open(output_filename, 'a+', encoding="utf-8") as f:
            tfidf_vectorizer = TfidfVectorizer(max_df=max_df, max_features=2000000,
                                               min_df=min_df, stop_words="english",
                                               use_idf=False, ngram_range=(1, 1))
            tfidf_matrix = tfidf_vectorizer.fit_transform(df_i['processedReviews'].values.astype('U'))
            terms = tfidf_vectorizer.get_feature_names()
            arr = []
            for i in range(0, tfidf_matrix.shape[0]):
                row = np.squeeze(tfidf_matrix[i].toarray())
                feats = top_tfidf_feats(row, terms, top_n)
                f.write(' '.join(str(e) for e in feats["tfidf"]))
                f.write("\n")


def top_n_keywords_average(output_filename, word2vec_model, text2kw_words, text2kw_coef, vector_size=256):
    words = pd.read_csv(text2kw_words, header=None, error_bad_lines=False, encoding='utf-8')
    coefs = genfromtxt(text2kw_coef, delimiter=' ')
    words = words[0]
    doc2vecs = []
    for i in range(0, len(words)):
        vec = [0 for k in range(vector_size)]
        for j in range(0, len(words[i].split())):
            if words[i].split()[j] in word2vec_model:
                vec += word2vec_model[words[i].split()[j]]*coefs[i][j]
        doc2vecs.append(vec)
    # Assuming res is a list of lists
    try:
        os.remove(output_filename)
    except:
        pass
    with open(output_filename, "a+", encoding="utf-8") as output:
        writer = csv.writer(output, lineterminator='\n')
        writer.writerows(doc2vecs)


def top_n_keywords_concat(label, word2vec_model, text2kw_words, text2kw_coef, vector_size=256):
    words = pd.read_csv(text2kw_words, header=None, error_bad_lines=False, encoding='utf-8')
    coefs = genfromtxt(text2kw_coef, delimiter=' ')
    words = words[0]
    print(words[0].split())
    labels = []
    X = np.zeros((len(words), 20, vector_size))  # , dtype=K.floatx())
    for i in range(0, len(words)):
        for j in range(0, len(words[i].split())):
            if words[i].split()[j] in word2vec_model.wv:
                print(i)
                print(j)
                print(words[i].split()[j])
                X[i, j, :] = word2vec_model[words[i].split()[j]]*coefs[i][j]
        labels.append(label)
    return X, labels


def top_n_keywords_concat_predict(word2vec_model, text2kw_words, text2kw_coef, vector_size=256):
    words = pd.read_csv(text2kw_words, header=None, error_bad_lines=False, encoding='utf-8')
    coefs = genfromtxt(text2kw_coef, delimiter=' ')
    words = words[0]
    X = np.zeros((len(words), 20, vector_size))  # , dtype=K.floatx())
    for i in range(0, len(words)):
        for j in range(0, len(words[i].split())):
            if words[i].split()[j] in word2vec_model:
                X[i, j, :] = word2vec_model[words[i].split()[j]]
    return X


def ElbowMethod(path, data, n):
    X = pd.read_csv(data, error_bad_lines=False, encoding='utf-8')
    distorsions = []
    for k in range(1, n):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X)
        distorsions.append(kmeans.inertia_)

    fig = plt.figure(figsize=(15, 5))
    plt.plot(range(1, n), distorsions, 'bo-')
    plt.grid(True)
    plt.ylabel("Square Root Error")
    plt.xlabel("Number of Clusters")
    plt.title('Elbow curve')
    plt.savefig(path+"ElbowCurve.png")
    plt.show(block=False)
    plt.close(fig)


def flatten(processedReviews_file, max_df, directory):
    extract_tfidf_keywords(directory+r"\keywords.csv", processedReviews_file, 20, max_df)
    extract_tfidf_coefs(directory+r"\coefs.csv", processedReviews_file, 20, max_df)
    model = Word2Vec.load("Word2vVec.model")
    top_n_keywords_average(directory+r"\text2kw_top_n_keywords.csv", model, directory+r"\keywords.csv",
                           directory+r"\coefs.csv", vector_size=256)
    ElbowMethod(directory, directory+r"\text2kw_top_n_keywords.csv", 10)

    results = ['coefs.csv', 'ElbowCurve.png', 'keywords.csv', 'text2kw_top_n_keywords.csv']
    onlyfiles = [f for f in listdir(directory) if isfile(join(directory, f))]
    intersection = set(results) & set(onlyfiles)
    if len(intersection) == len(results):
        return "Data Flattened with success"
    else:
        return "error"


def clustering(outputfile_name, filename_data, filename_doc2vec, num_clusters):
    data = pd.read_csv(filename_data, error_bad_lines=False, encoding='utf-8')
    doc2vec = pd.read_csv(filename_doc2vec, header=None, error_bad_lines=False, encoding='utf-8')

    kclusterer = KMeansClusterer(num_clusters, distance=nltk.cluster.util.cosine_distance,
                                 avoid_empty_clusters=True, repeats=10)
    map = np.asarray([(np.asarray(f[1].data), f[0]) for f in doc2vec.iterrows() if any(np.asarray(f[1].data)) != 0])
    indexes = [map[i][1] for i in range(len(map)-1)]
    vectors = np.asarray([np.asarray(map[i][0]) for i in range(len(map)-1)])

    clusters = kclusterer.cluster(vectors, assign_clusters=True)

    data_df_i = pd.DataFrame(list(
        zip([data["Text"][i] for i in indexes], [data["processedReviews"][i] for i in indexes],
            [data["Date"][i] for i in indexes], [data["Sentiments"][i] for i in indexes],
            [data["Country"][i] for i in indexes], clusters)),
        columns=['Text', 'processedReviews', 'Date', 'Sentiments', 'Country', 'Cluster'])
    try:
        os.remove(outputfile_name)
    except:
        pass

    print(filename_doc2vec)
    try:
        os.remove(filename_doc2vec)
    except:
        pass
    try:
        os.remove(filename_doc2vec.replace("text2kw_top_n_keywords.csv", "keywords.csv"))
    except:
        pass
    try:
        os.remove(filename_doc2vec.replace("text2kw_top_n_keywords.csv", "coefs.csv"))
    except:
        pass
    with open(outputfile_name, 'w+', encoding="utf-8") as file:
        data_df_i.to_csv(file, header=True, index=False, index_label=False)
        file.close()


def Plot_Clusters_Kmeans(outputfile_name, nbr_clusters, path):
    list_stop_words = [stemmer.stem(stopWords[i]) for i in range(len(stopWords))]

    text2kw_clusters = pd.read_csv(outputfile_name, error_bad_lines=False, encoding='utf-8')
    text2kw_clusters['Cluster'] = text2kw_clusters['Cluster'].astype(str)
    corpus = (st.CorpusFromPandas(text2kw_clusters, category_col='Cluster', text_col='Text', nlp=nlp).build(). \
              remove_terms(ENGLISH_STOP_WORDS, ignore_absences=True).
              get_unigram_corpus().compact(st.ClassPercentageCompactor(term_count=2,
                                                                       term_ranker=st.OncePerDocFrequencyRanker)))

    for i in range(nbr_clusters):
        directory = path+r"\\"+str(i)+r"\\"
        try:
            os.mkdir(directory)
        except FileExistsError:
            pass

        html = st.produce_scattertext_explorer(corpus, category=str(i), category_name=str(i)+" Category",
                                               not_category_name='Other Categories',
                                               metadata=text2kw_clusters['Date'],
                                               minimum_term_frequency=50)
        filename = directory+str(i)+"_Category-VS-other categories.html"
        open(filename, 'wb+').write(html.encode('utf-8'))

    text2kw_clusters = pd.read_csv(outputfile_name, error_bad_lines=False, encoding='utf-8')
    text2kw_clusters['Cluster'] = text2kw_clusters['Cluster'].astype(str)
    text2kw_clusters['Text'] = text2kw_clusters['Text'].apply(nlp)
    corpus = (st.CorpusFromParsedDocuments(text2kw_clusters, category_col='Cluster', parsed_col='Text')).build(). \
        remove_terms(ENGLISH_STOP_WORDS, ignore_absences=True)
    for i in range(nbr_clusters):
        directory = path+r"\\"+str(i)+r"\\"

        m = text2kw_clusters[text2kw_clusters["Cluster"] == str(i)]
        liste = [word_tokenize(str(x)) for x in m["processedReviews"] if not stemmer.stem(str(x)) in list_stop_words]
        words = []
        for j in range(len(liste)):
            for k in range(len(liste[j])):
                if not (liste[j][k] in list_stop_words):
                    try:
                        words.append(liste[j][k])
                    except:
                        pass

        counter = collections.Counter(words)
        c = counter.most_common()
        html = word_similarity_explorer_gensim(corpus, category=str(i), category_name=str(i)+" Category",
                                               not_category_name='Other Categories',
                                               minimum_term_frequency=int(text2kw_clusters.shape[0]*0.005),
                                               target_term=stemmer.stem(c[0][0]),
                                               # pmi_threshold_coefficient=4,
                                               width_in_pixels=1000,
                                               metadata=text2kw_clusters['Date'],
                                               word2vec=model,
                                               max_p_val=0.05,
                                               save_svg_button=True)
        filename = directory+str(i)+"_w2v_Category-VS-other categories.html"
        open(filename, 'wb+').write(html.encode('utf-8'))

    for i in range(nbr_clusters):
        directory = path+r"\\"+str(i)+r"\\"

        text2kw_clusters = pd.read_csv(outputfile_name, error_bad_lines=False, encoding='utf-8')
        text2kw_clusters['Cluster'] = text2kw_clusters['Cluster'].astype(str)
        text2kw_clusters['Sentiments'] = text2kw_clusters['Sentiments'].astype(str)
        text2kw_clusters['Date'] = text2kw_clusters['Date'].astype(str)
        text2kw_clusters = text2kw_clusters.loc[text2kw_clusters["Cluster"] == str(i)]
        text2kw_clusters = text2kw_clusters.loc[text2kw_clusters["Sentiments"] != "neutral"]
        corpus = (st.CorpusFromPandas(text2kw_clusters, category_col='Sentiments', text_col='Text', nlp=nlp).build().
                  remove_terms(ENGLISH_STOP_WORDS, ignore_absences=True))
        html = st.produce_scattertext_explorer(corpus, category="positive", category_name="Positive Verbatims",
                                               not_category_name='Negative Verbatims',
                                               metadata=text2kw_clusters['Date'],
                                               minimum_term_frequency=int(text2kw_clusters.shape[0]*0.005))
        filename = directory+str(i)+"_Positive_Category-VS-Negative_Category.html"
        open(filename, 'wb+').write(html.encode('utf-8'))

    for i in range(nbr_clusters):
        directory = path+r"\\"+str(i)+r"\\"

        text2kw_clusters = pd.read_csv(outputfile_name, error_bad_lines=False, encoding='utf-8')
        text2kw_clusters['Cluster'] = text2kw_clusters['Cluster'].astype(str)
        text2kw_clusters['Sentiments'] = text2kw_clusters['Sentiments'].astype(str)
        text2kw_clusters = text2kw_clusters.loc[text2kw_clusters["Cluster"] == str(i)]
        text2kw_clusters = text2kw_clusters.loc[text2kw_clusters["Sentiments"] != "neutral"]
        text2kw_clusters['Text'] = text2kw_clusters['Text'].apply(nlp)
        liste = [word_tokenize(str(x)) for x in text2kw_clusters["processedReviews"] if
                 not stemmer.stem(str(x)) in list_stop_words]
        words = []
        for j in range(len(liste)):
            for k in range(len(liste[j])):
                if not (liste[j][k] in list_stop_words):
                    try:
                        words.append(liste[j][k])
                    except:
                        pass
        counter = collections.Counter(words)
        c = counter.most_common()

        corpus = (st.CorpusFromParsedDocuments(text2kw_clusters, category_col='Sentiments', parsed_col='Text')).build(). \
            remove_terms(ENGLISH_STOP_WORDS, ignore_absences=True)
        html = word_similarity_explorer_gensim(corpus, category="positive", category_name="Positive Verbatims",
                                               not_category_name='Negative Verbatims',
                                               minimum_term_frequency=int(text2kw_clusters.shape[0]*0.005),
                                               target_term=stemmer.stem(c[0][0]),
                                               # pmi_threshold_coefficient=4,
                                               width_in_pixels=1000,
                                               metadata=text2kw_clusters['Date'],
                                               word2vec=model,
                                               max_p_val=0.05,
                                               save_svg_button=True)
        filename = directory+str(i)+"_w2v__Positive_Category-VS-Negative_Category.html"
        open(filename, 'wb+').write(html.encode('utf-8'))


def wordcloud_cluster_byIds_Kmeans(path, file_name, nbr_clusters):
    list_stop_words = [stemmer.stem(stopWords[i]) for i in range(len(stopWords))]
    nclust = nbr_clusters

    for i in range(nclust):
        directory = path+r"\\"+str(i)+r"\\"
        data = pd.read_csv(file_name, error_bad_lines=False, encoding='utf-8')
        data_i = data.loc[data["Cluster"] == i]
        liste = [word_tokenize(str(x)) for x in data_i["processedReviews"] if
                 not stemmer.stem(str(x)) in list_stop_words]

        words = ''
        for j in range(len(liste)):
            for k in range(len(liste[j])):
                if liste[j][k] not in list_stop_words:
                    try:
                        words = words+liste[j][k]+' '
                    except:
                        pass

        wordcloud = WordCloud(
            background_color='white',
            max_words=100,
            max_font_size=40,
            scale=3,
            random_state=1  # chosen at random by flipping a coin; it was heads
        ).generate(words)

        plt.tight_layout()
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.rcParams['figure.figsize'] = 10, 8
        plt.savefig(str(directory)+str(i)+"_WordCloud.png")
        plt.close()


def Plot_Time_Series_Kmeans(path, file_name, nbr_clusters):
    list_stop_words = [stemmer.stem(stopWords[i]) for i in range(len(stopWords))]
    all = pd.read_csv(file_name, encoding='utf8')
    colors_1 = [plt.cm.Blues, plt.cm.Reds, plt.cm.Greens, plt.cm.Purples, plt.cm.YlOrBr, plt.cm.Greys,
                plt.cm.YlGn, plt.cm.PuRd, plt.cm.bone, plt.cm.spring, plt.cm.afmhot, plt.cm.BuGn]

    dates = []
    for i in range(all.shape[0]):
        dates = []
        for i in range(all.shape[0]):
            try:
                if (isNaN(all["Date"][i]) == False):
                    dates.append(i)
            except:
                pass
    dates_data = all.loc[dates]
    dates_data.index = pd.to_datetime(dates_data["Date"])
    line_width = 1.5
    fig, ax1 = plt.subplots()
    ax1.legend(loc=1)
    fig.legend(loc=1)
    for key in range(int(nbr_clusters)):
        m = dates_data[dates_data["Cluster"] == key]
        liste = [word_tokenize(x) for x in m["processedReviews"] if not stemmer.stem(x) in list_stop_words]
        c = ["group 1", "group 2", "group 3", "group 4", "group 5", "group 6", "group 7", "group 8", "group 9",
             "group 10"]
        m = dates_data[dates_data["Cluster"] == key].resample("15d").count()
        with plt.style.context('bmh'):
            # plot command goes here
            m.plot(x=m.index, y='Cluster', ax=ax1, lw=line_width, label=str(c[key]),
                   color=colors_1[key](0.8))
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Number of Mentions', color='black')
    with plt.style.context('bmh'):
        plt.rcParams['figure.figsize'] = 11, 9
        plt.savefig(str(path)+key+"/Mentions-VS-Time.png")
        plt.close(fig)


def main(path, cat):
    # create a data frame for each group (data + doc2vec)
    try:
        clusters = pd.read_csv(path+r"\Outputs\DataSets\Data_clustered.csv", error_bad_lines=False,
                               encoding='utf-8')
        clusters['Cluster'] = clusters['Cluster'].astype(str)
        cluster = clusters.loc[clusters["Cluster"] == str(cat)]
        cluster.index = range(len(cluster.index))

        directory = path+r"\Outputs\Classes\\"+cat
        file_name = path+r"\Outputs\Classes\\"+cat+r"\\"+cat+".csv"
        outputfile_name = path+r"\Outputs\Classes\\"+cat+r"\\"+cat+"_clustered.csv"

        cluster.to_csv(file_name, header=True, index=False, index_label=False)
        del cluster
        flatten(file_name, 0.7, directory+r"\\")

        return directory, file_name, outputfile_name

    except:
        pass

import pandas as pd
import numpy as np
import csv
import nltk
from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer('english')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.cluster import KMeansClusterer
import spacy

spacy.prefer_gpu()
nlp = spacy.load("en")
from numpy import genfromtxt
from nltk.corpus import stopwords
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS

ENGLISH_STOP_WORDS = list(ENGLISH_STOP_WORDS)
ENGLISH_STOP_WORDS.append("hello")
ENGLISH_STOP_WORDS.append("evening")
ENGLISH_STOP_WORDS.append("thank")
ENGLISH_STOP_WORDS.append("thanks")
ENGLISH_STOP_WORDS.append("love")
ENGLISH_STOP_WORDS.append("club")
ENGLISH_STOP_WORDS.append("hi")
ENGLISH_STOP_WORDS.append("forum")
ENGLISH_STOP_WORDS = tuple(ENGLISH_STOP_WORDS)

stopWords = stopwords.words('english')
stopWords.append(["com", "at_user", "pic", "twitter"])
from math import ceil
import scattertext as st
import spacy
from nltk.sentiment.vader import SentimentIntensityAnalyzer
# nltk.download('vader_lexicon')
from nltk.tokenize import word_tokenize
import collections
from pylab import rcParams

import itertools
from gensim.models import Word2Vec
model = Word2Vec.load("/home/marwen/PycharmProjects/Renault/ZOE/EVs_word2vec(128).model")

import spacy
from scattertext import word_similarity_explorer_gensim

nlp = spacy.load("en")

technical_words = []

technical_words.append([item[0] for item in model.wv.most_similar(positive=[stemmer.stem("battery")], topn=200)])
technical_words.append([item[0] for item in model.wv.most_similar(positive=[stemmer.stem("charge")], topn=200)])
technical_words.append([item[0] for item in model.wv.most_similar(positive=[stemmer.stem("plug")], topn=200)])
technical_words.append([item[0] for item in model.wv.most_similar(positive=[stemmer.stem("brakes")], topn=200)])
technical_words.append([item[0] for item in model.wv.most_similar(positive=[stemmer.stem("engine")], topn=200)])
technical_words.append([item[0] for item in model.wv.most_similar(positive=[stemmer.stem("gearbox")], topn=200)])
technical_words.append([item[0] for item in model.wv.most_similar(positive=[stemmer.stem("equipements")], topn=200)])
technical_words.append([item[0] for item in model.wv.most_similar(positive=[stemmer.stem("feature")], topn=200)])
technical_words.append([item[0] for item in model.wv.most_similar(positive=[stemmer.stem("bluetooth")], topn=200)])
technical_words.append([item[0] for item in model.wv.most_similar(positive=[stemmer.stem("lights")], topn=200)])
technical_words.append([item[0] for item in model.wv.most_similar(positive=[stemmer.stem("camera")], topn=200)])
technical_words.append([item[0] for item in model.wv.most_similar(positive=[stemmer.stem("doors")], topn=200)])
technical_words.append([item[0] for item in model.wv.most_similar(positive=[stemmer.stem("interior")], topn=200)])
technical_words.append([item[0] for item in model.wv.most_similar(positive=[stemmer.stem("seats")], topn=200)])
technical_words.append([item[0] for item in model.wv.most_similar(positive=[stemmer.stem("price")], topn=200)])
technical_words.append([item[0] for item in model.wv.most_similar(positive=[stemmer.stem("cost")], topn=200)])
technical_words.append([item[0] for item in model.wv.most_similar(positive=[stemmer.stem("tax")], topn=200)])

technical_words.append(stemmer.stem("battery"))
technical_words.append(stemmer.stem("charge"))
technical_words.append(stemmer.stem("brakes"))
technical_words.append(stemmer.stem("engine"))
technical_words.append(stemmer.stem("gearbox"))
technical_words.append(stemmer.stem("camera"))
technical_words.append(stemmer.stem("lights"))
technical_words.append(stemmer.stem("equipements"))
technical_words.append(stemmer.stem("carnet"))
technical_words.append(stemmer.stem("car-net"))
technical_words.append(stemmer.stem("doors"))
technical_words.append(stemmer.stem("interior"))
technical_words.append(stemmer.stem("seats"))
technical_words.append(stemmer.stem("price"))

technical_words = list(itertools.chain(*technical_words))

def isNaN(num):
    return num != num


def top_tfidf_feats(row, features, top_n=25):
    topn_ids = np.argsort(row)[::-1][:top_n]
    top_feats = [(features[i], row[i]) for i in topn_ids]
    df = pd.DataFrame(top_feats)
    df.columns = ['feature', 'tfidf']
    return df


def extract_tfidf_keywords(output_filename, input_filename, top_n, max_df=1.0, min_df=0.0):
    df = pd.read_csv(input_filename, sep=',', chunksize=10000)
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
                f.write(' '.join(str(e) for e in feats["feature"]))
                f.write("\n")


def extract_tfidf_coefs(output_filename, input_filename, top_n, max_df=1.0, min_df=0.0):
    df = pd.read_csv(input_filename, sep=',', chunksize=10000)
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


"""
    step two:
    word2vec representation
"""
# word2vec_model = gensim.models.Word2Vec(text2kw, size=256, window=5, min_count=5, workers=4, iter=50)
# model.save("word2vec.model")
# model = Word2Vec.load("word2vec.model")
"""
similar_words = {search_term: [item[0] for item in word2vec_model.wv.most_similar([search_term], topn=10)]
                 for search_term in [stemmer.stem("nissan")]}
print(similar_words)
"""


def top_n_keywords_average(output_filename, word2vec_model, text2kw_words, text2kw_coef, vector_size=256):
    words = pd.read_csv(text2kw_words, header=None, error_bad_lines=False, encoding='utf-8')
    coefs = genfromtxt(text2kw_coef, delimiter=' ')
    words = words[0]
    doc2vecs = []
    for i in range(0, len(words)):
        vec = [0 for k in range(vector_size)]
        for j in range(0, len(words[i].split())):
            if words[i].split()[j] in word2vec_model:
                vec += word2vec_model[words[i].split()[j]] * coefs[i][j]
        doc2vecs.append(vec)
    # Assuming res is a list of lists
    with open(output_filename, "a+", encoding="utf-8") as output:
        writer = csv.writer(output, lineterminator='\n')
        writer.writerows(doc2vecs)


def top_n_keywords_concat(label, word2vec_model, text2kw_words, text2kw_coef, vector_size=128):
    words = pd.read_csv(text2kw_words, header=None, error_bad_lines=False, encoding='utf-8')
    coefs = genfromtxt(text2kw_coef, delimiter=' ')
    words = words[0]
    labels = []
    X = np.zeros((len(words), 20, vector_size))#, dtype=K.floatx())
    for i in range(0, len(words)):
        for j in range(0, len(words[i].split())):
            if words[i].split()[j] in word2vec_model:
                X[i, j, :] = word2vec_model[words[i].split()[j]] * coefs[i][j]
        labels.append(label)
    return X, labels


def top_n_keywords_concat_predict(word2vec_model, text2kw_words, text2kw_coef, vector_size=128):
    words = pd.read_csv(text2kw_words, header=None, error_bad_lines=False, encoding='utf-8')
    coefs = genfromtxt(text2kw_coef, delimiter=' ')
    words = words[0]
    X = np.zeros((len(words), 20, vector_size))#, dtype=K.floatx())
    for i in range(0, len(words)):
        for j in range(0, len(words[i].split())):
            if words[i].split()[j] in word2vec_model:
                X[i, j, :] = word2vec_model[words[i].split()[j]]
    return X
"""
    # Assuming res is a list of lists
    with open(output_filename, "a+", encoding="utf-8") as output:
        writer = csv.writer(output, lineterminator='\n')
        writer.writerows(X)
    # Write the array to disk
    with open(output_filename, "a+", encoding="utf-8") as outfile:
        # I'm writing a header here just for the sake of readability
        # Any line starting with "#" will be ignored by numpy.loadtxt
        outfile.write('# Array shape: {0}\n'.format(X.shape))

        # Iterating through a ndimensional array produces slices along
        # the last axis. This is equivalent to data[i,:,:] in this case
        for data_slice in X:
            # The formatting string indicates that I'm writing out
            # the values in left-justified columns 7 characters in width
            # with 2 decimal places.
            np.savetxt(outfile, data_slice, fmt='%-7.8f')

            # Writing out a break to indicate different slices...
            outfile.write('# New slice\n')
"""

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
    plt.savefig(path + "ElbowCurve.png")
    plt.show(block=False)
    plt.close(fig)


from gensim.models import Word2Vec
"""
def clustering(outputfile_name, filename_data, filename_doc2vec, num_clusters):
    chunksize = 1000000
    list_words = ["battery", "brakes", "doors", "engine-gearbox", "equipements", "doors-interior-seats", "lights"]
    model = Word2Vec.load("EVs_word2vec(128).model")
    data = pd.read_csv(filename_data, chunksize=chunksize, error_bad_lines=False, encoding='utf-8')


        clusters = kclusterer.cluster(vectors, assign_clusters=True)
        print(clusters)
        data_df_i = pd.DataFrame(list(
            zip(data_i["Reviews"], data_i["processedReviews"] ,
                data_i["Date"], data_i["Sentiments"], [list_words[i] for i in clusters])),
                                 columns=['Reviews', 'processedReviews', 'Date', 'Sentiments', 'Cluster'])

        with open(outputfile_name, 'w+', encoding="utf-8") as file:
            if i == 0:
                data_df_i.to_csv(file, header=True, index=False, index_label=False)
                file.close()
            else:
                data_df_i.to_csv(file, header=False, index=False, index_label=False)
                file.close()
"""

def Plot_Clusters_trained(filename_data, path):
    categories = ["seats"]
    nbr_clusters = len(categories)
    list_stop_words = ["zoe", "vehicle", "electric", "car", "ev"]
    list_stop_words = [stemmer.stem(list_stop_words[i]) for i in range(len(list_stop_words))]
    groups = categories
    subgroup_names = []
    subgroup_size = []

    for key in categories:
        text2kw_clusters = pd.read_csv(filename_data, error_bad_lines=False, encoding='utf-8')

        text2kw_clusters['Cluster'] = text2kw_clusters['Cluster'].astype(str)
        text2kw_clusters['Sentiments'] = text2kw_clusters['Sentiments'].astype(str)
        cluster = text2kw_clusters.loc[text2kw_clusters["Cluster"] == str(key)]

        positive = cluster.loc[cluster["Sentiments"] == "positive"]
        subgroup_names.append("Positive")
        subgroup_size.append(positive.shape[0])

        neutral = cluster.loc[cluster["Sentiments"] == "neutral"]
        subgroup_names.append("Neutral")
        subgroup_size.append(neutral.shape[0])

        negative = cluster.loc[cluster["Sentiments"] == "negative"]
        subgroup_names.append("Negative")
        subgroup_size.append(negative.shape[0])

    text2kw_clusters = pd.read_csv(filename_data, error_bad_lines=False, encoding='utf-8')

    count = text2kw_clusters.groupby("Cluster")["Cluster"].count()
    data = list(count)

    # Make data: I have 9 groups and 9*3 subgroups
    group_names = groups
    group_size = data

    # Create colors
    colors_1 = [plt.cm.Blues, plt.cm.Reds, plt.cm.Greens, plt.cm.Purples, plt.cm.Greys, plt.cm.Oranges, plt.cm.Wistia]

    # First Ring (outside)
    fig, ax = plt.subplots(figsize=(12, 7))
    plt.rcParams['figure.figsize'] = 12, 7

    ax.axis('equal')

    print(group_size)
    print(group_names)

    colors_2 = []
    for i in range(nbr_clusters):
        colors_2.append(colors_1[i](0.9))
    mypie, _ = ax.pie(group_size, radius=1.3, labels=group_names, labeldistance=1.05, colors=colors_2)
    plt.setp(mypie, width=0.3, edgecolor='white')

    colors_3 = []
    for i in range(nbr_clusters):
        colors_3.append(colors_1[i](0.8))
        colors_3.append(colors_1[i](0.6))
        colors_3.append(colors_1[i](0.4))
    # Second Ring (Inside)
    mypie2, _ = ax.pie(subgroup_size, radius=1.3 - 0.3, labels=subgroup_names, labeldistance=0.65, colors=colors_3,
                       rotatelabels=True, startangle=360, counterclock=True)
    plt.setp(mypie2, width=0.4, edgecolor='white')
    plt.margins(0, 0)

    plt.savefig(path + "Sentiments.png", optimize=True)
    plt.rcParams['figure.figsize'] = 14, 7
    plt.close(fig)

    text2kw_clusters = pd.read_csv(filename_data, error_bad_lines=False, encoding='utf-8')
    text2kw_clusters['Cluster'] = text2kw_clusters['Cluster'].astype(str)
    corpus = (st.CorpusFromPandas(text2kw_clusters, category_col='Cluster', text_col='Reviews', nlp=nlp).build().
              remove_terms(ENGLISH_STOP_WORDS, ignore_absences=True).
              get_unigram_corpus().compact(st.ClassPercentageCompactor(term_count=2, term_ranker=st.OncePerDocFrequencyRanker)))
    for i in categories:
        html = st.produce_scattertext_explorer(corpus, category=str(i), category_name=str(i) + " Category",
                                               not_category_name='Other Categories',
                                               metadata=text2kw_clusters['Date'],
                                               minimum_term_frequency=int(text2kw_clusters.shape[0]*0.01))
        filename = path + str(i) + "_seats-VS-other categories.html"
        open(filename, 'wb+').write(html.encode('utf-8'))


    text2kw_clusters = pd.read_csv(filename_data, error_bad_lines=False, encoding='utf-8')
    text2kw_clusters['Cluster'] = text2kw_clusters['Cluster'].astype(str)
    text2kw_clusters['Reviews'] = text2kw_clusters['Reviews'].apply(nlp)
    corpus = (st.CorpusFromParsedDocuments(text2kw_clusters, category_col='Cluster', parsed_col='Reviews')).build(). \
        remove_terms(ENGLISH_STOP_WORDS, ignore_absences=True)
    for i in categories:
        html = word_similarity_explorer_gensim(corpus, category=str(i), category_name=str(i) + " Category",
                                               not_category_name='Other Categories',
                                               minimum_term_frequency=int(text2kw_clusters.shape[0] * 0.005),
                                               target_term=stemmer.stem(str(i)),
                                               # pmi_threshold_coefficient=4,
                                               width_in_pixels=1000,
                                               metadata=text2kw_clusters['Date'],
                                               word2vec=model,
                                               max_p_val=0.05,
                                               save_svg_button=True)
        filename = path + str(i) + "_w2v_seats-VS-other categories.html"
        open(filename, 'wb+').write(html.encode('utf-8'))

    for i, classe in enumerate(categories):
        cat = ["seats"]
        text2kw_clusters = pd.read_csv(filename_data, error_bad_lines=False, encoding='utf-8')
        text2kw_clusters['Cluster'] = text2kw_clusters['Cluster'].astype(str)
        text2kw_clusters['Sentiments'] = text2kw_clusters['Sentiments'].astype(str)
        text2kw_clusters = text2kw_clusters.loc[text2kw_clusters["Cluster"] == str(classe)]
        text2kw_clusters = text2kw_clusters.loc[text2kw_clusters["Sentiments"] != "neutral"]
        text2kw_clusters['Reviews'] = text2kw_clusters['Reviews'].apply(nlp)
        corpus = (st.CorpusFromParsedDocuments(text2kw_clusters, category_col='Sentiments', parsed_col='Reviews')).build(). \
            remove_terms(ENGLISH_STOP_WORDS, ignore_absences=True)
        html = word_similarity_explorer_gensim(corpus, category="positive", category_name="Positive Verbatims",
                                               not_category_name='Negative Verbatims',
                                               minimum_term_frequency=int(text2kw_clusters.shape[0] * 0.005),
                                               target_term=stemmer.stem(str(cat[i])),
                                               # pmi_threshold_coefficient=4,
                                               width_in_pixels=1000,
                                               metadata=text2kw_clusters['Date'],
                                               word2vec=model,
                                               max_p_val=0.05,
                                               save_svg_button=True)
        filename = path + str(classe) + "_w2v__seats_Category-VS-Negative_Category.html"
        open(filename, 'wb+').write(html.encode('utf-8'))


    for i in categories:
        text2kw_clusters = pd.read_csv(filename_data , error_bad_lines=False, encoding='utf-8')
        text2kw_clusters['Cluster'] = text2kw_clusters['Cluster'].astype(str)
        text2kw_clusters['Sentiments'] = text2kw_clusters['Sentiments'].astype(str)
        text2kw_clusters = text2kw_clusters.loc[text2kw_clusters["Cluster"] == str(i)]
        text2kw_clusters = text2kw_clusters.loc[text2kw_clusters["Sentiments"] != "neutral"]
        corpus = st.CorpusFromPandas(text2kw_clusters, category_col='Sentiments', text_col='Reviews', nlp=nlp).build().\
            remove_terms(ENGLISH_STOP_WORDS, ignore_absences=True)
        html = st.produce_scattertext_explorer(corpus, category="positive", category_name="Positive Verbatims",
                                               not_category_name='Negative Verbatims',
                                               metadata=text2kw_clusters['Date'],
                                               minimum_term_frequency=int(cluster.shape[0]*0.01))
        filename = path + str(i) + "_Positive_Category-VS-Negative_Category.html"
        open(filename, 'wb+').write(html.encode('utf-8'))



def Plot_Clusters_by_country(filename_data, path):
    import matplotlib.pyplot as plt

    text2kw_clusters = pd.read_csv(filename_data, error_bad_lines=False, encoding='utf-8')
    categories = ["seats"]
    countries = set(text2kw_clusters["Lang"])
    liste = {'seats': 0}

    for country in countries:
        dict = {}
        for categorie in categories:
            text2kw_clusters = pd.read_csv(filename_data, error_bad_lines=False, encoding='utf-8')

            text2kw_clusters['Lang'] = text2kw_clusters['Lang'].astype(str)
            cluster = text2kw_clusters.loc[text2kw_clusters["Lang"] == str(country)]

            cluster['Cluster'] = cluster['Cluster'].astype(str)
            cluster = cluster.loc[cluster["Cluster"] == str(categorie)]

            cluster['Sentiments'] = cluster['Sentiments'].astype(str)

            liste[categorie] = [cluster.loc[cluster["Sentiments"] == "positive"].shape[0],
                          cluster.loc[cluster["Sentiments"] == "neutral"].shape[0],
                          cluster.loc[cluster["Sentiments"] == "negative"].shape[0]]

        raw_data = liste
        df = pd.DataFrame(raw_data).T
        df.columns = ["Positive", "Neutral", "Negative"]

        # Create a figure with a single subplot
        f, ax = plt.subplots(1, figsize=(7, 5))
        f.subplots_adjust(bottom=0.3, left=0.2)

        # Set bar width at 1
        bar_width = 0.7

        # positions of the left bar-boundaries
        bar_l = [i for i in range(len(df['Positive']))]

        # positions of the x-axis ticks (center of the bars as bar labels)
        tick_pos = [i + (bar_width / 2) for i in bar_l]

        # Create the total score for each participant
        totals = [i + j + k for i, j, k in zip(df['Positive'], df['Neutral'], df['Negative'])]

        # Create the percentage of the total score the pre_score value for each participant was
        pre_rel = [i / j * 100 for i, j in zip(df['Positive'], totals)]

        # Create the percentage of the total score the mid_score value for each participant was
        mid_rel = [i / j * 100 for i, j in zip(df['Neutral'], totals)]

        # Create the percentage of the total score the post_score value for each participant was
        post_rel = [i / j * 100 for i, j in zip(df['Negative'], totals)]

        # Create a bar chart in position bar_1
        ax.bar(bar_l,
               # using pre_rel data
               pre_rel,
               # labeled
               label='Positive',
               # with alpha
               alpha=0.6,
               # with color
               color='red',
               # with bar width
               width=bar_width,
               # with border color
               edgecolor='white'
               )

        # Create a bar chart in position bar_1
        ax.bar(bar_l,
               # using mid_rel data
               mid_rel,
               # with pre_rel
               bottom=pre_rel,
               # labeled
               label='Neutral',
               # with alpha
               alpha=0.6,
               # with color
               color='#3C5F5A',
               # with bar width
               width=bar_width,
               # with border color
               edgecolor='white'
               )

        # Create a bar chart in position bar_1
        ax.bar(bar_l,
               # using post_rel data
               post_rel,
               # with pre_rel and mid_rel on bottom
               bottom=[i + j for i, j in zip(pre_rel, mid_rel)],
               # labeled
               label='Post Score',
               # with alpha
               alpha=0.6,
               # with color
               color='#019600',
               # with bar width
               width=bar_width,
               # with border color
               edgecolor='white'
               )

        plt.xticks(tick_pos, categories, fontsize=10)
        ax.set_ylabel("Percentage", fontsize=10)
        ax.set_xlabel("", fontsize=10)

        plt.xlim([min(tick_pos) - 2 * bar_width, max(tick_pos) + bar_width])
        plt.ylim(-10, 110)
        plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right', fontsize=12)

        plt.show()
        plt.savefig(str(country) + "_barplot.png")
        plt.close()


def wordcloud_cluster_byIds(filename):
    categories = ["seats"]
    list_stop_words = ["e-golf", "vehicle", "electric", "e", "golf", "volkswagen", "car", "wagen", "ev", "volks", "vw", "egolf",
                       "n't"]
    list_stop_words = [stemmer.stem(list_stop_words[i]) for i in range(len(list_stop_words))]
    data = pd.read_csv(filename, error_bad_lines=False, encoding='utf-8')

    for i, key in enumerate(categories):
        data = pd.read_csv(filename, error_bad_lines=False, encoding='utf-8')
        data_i = data.loc[data["Cluster"] == key]

        try:
            liste = [word_tokenize(str(x)) for x in data_i["processedReviews"] if
                     not stemmer.stem(str(x)) in list_stop_words]
            words = ''
            for j in range(len(liste)):
                for k in range(len(liste[j])):
                    if liste[j][k] not in list_stop_words:
                        try:
                            words = words + liste[j][k] + ' '
                        except:
                            pass
            wordcloud = WordCloud(
                background_color='white',
                max_words=100,
                max_font_size=40,
                scale=3,
                random_state=1  # chosen at random by flipping a coin; it was heads
            ).generate(words)

            #plt.rcParams['figure.figsize'] = 8, 8
            plt.tight_layout()
            plt.imshow(wordcloud)
            plt.axis("off")
            plt.savefig(str(key) + "_WordCloud.png")
            plt.close()
        except:
            pass

    for i, key in enumerate(categories):
        data = pd.read_csv(filename, error_bad_lines=False, encoding='utf-8')
        liste = []
        for sent in ["negative", "positive"]:
            data_i = data.loc[data["Cluster"] == key]
            data_i = data_i.loc[data["Sentiments"] == sent]
            try:
                liste = [word_tokenize(str(x)) for x in data_i["processedReviews"] if
                         not stemmer.stem(str(x)) in list_stop_words]

                words = ''
                for j in range(len(liste)):
                    for k in range(len(liste[j])):
                        if liste[j][k] not in list_stop_words and liste[j][k] in technical_words:
                            try:
                                words = words + liste[j][k] + ' '
                            except:
                                pass

                wordcloud = WordCloud(
                    background_color='white',
                    max_words=100,
                    max_font_size=40,
                    scale=3,
                    random_state=1  # chosen at random by flipping a coin; it was heads
                ).generate(words)

                plt.rcParams['figure.figsize'] = 13, 10
                plt.tight_layout()
                plt.imshow(wordcloud, interpolation="bilinear")
                plt.axis("off")
                plt.savefig(str(key) + "_" + str(sent) + "_WordCloud.png")
                plt.close()
            except:
                pass


def score_Review(Reviews):
    df_dict = {}
    for i, response in enumerate(Reviews):
        scorer = SentimentIntensityAnalyzer()
        scores = scorer.polarity_scores(response)
        df_dict[i] = scores

    df = pd.DataFrame.from_dict(df_dict, orient='columns')
    df = df.T

    diff_score = df['pos'] - 2 * df['neg']
    for i in range(len(diff_score)):
        if diff_score[i] > 0.05:
            diff_score[i] = "positive"
        elif diff_score[i] < -0.05:
            diff_score[i] = "negative"
        else:
            diff_score[i] = "neutral"
    diff_score.columns = ["Sentiments"]
    sentiments = pd.concat([df, diff_score], axis=1)
    return sentiments


def Plot_Time_Series(filename, list_stop_words=[]):
    categories = ["seats"]
    colors_1 = [plt.cm.Blues, plt.cm.Reds, plt.cm.Greens, plt.cm.Purples, plt.cm.YlOrBr, plt.cm.Greys,
                plt.cm.YlGn, plt.cm.PuRd, plt.cm.bone, plt.cm.spring, plt.cm.afmhot, plt.cm.BuGn]

    all = pd.read_csv(filename, encoding='utf8')

    dates = []
    for i in range(all.shape[0]):
        try:
            if (isNaN(all["Date"][i]) == False) and (int(all["Date"][i].split("-")[0]) > int(2013)):
                dates.append(i)
        except:
            pass

    dates_data = all.loc[dates]
    dates_data.index = pd.to_datetime(dates_data["Date"])
    line_width = 1.5
    fig, ax1 = plt.subplots()
    ax1.legend(loc=1)
    fig.legend(loc=1)
    for i, key in enumerate(categories):
        m = dates_data[dates_data["Cluster"] == key].resample("15d").count()
        with plt.style.context('bmh'):
            # plot command goes here
            m.plot(x=m.index, y='Cluster', ax=ax1, lw=line_width, label=str(key),
                   color=colors_1[i](0.8), figsize=(12, 8))
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Number of Mentions', color='black')
    with plt.style.context('bmh'):
        plt.rcParams['figure.figsize'] = 7, 5
        plt.savefig("Seats_Mentions-VS-Time.png")
        plt.close(fig)
######################################################################################



def clustering(outputfile_name, filename_data, filename_doc2vec, num_clusters):
    data = pd.read_csv(filename_data,  error_bad_lines=False, encoding='utf-8')
    doc2vec = pd.read_csv(filename_doc2vec, header=None, error_bad_lines=False, encoding='utf-8')

    indexes = []
    kclusterer = KMeansClusterer(num_clusters, distance=nltk.cluster.util.cosine_distance,
                                 avoid_empty_clusters=True, repeats=10)
    map = np.asarray(
        [(np.asarray(f[1].data), f[0]) for f in doc2vec.iterrows() if any(np.asarray(f[1].data)) != 0])
    indexes = [map[i][1] for i in range(len(map)-1)]
    print(len(indexes))
    print(data.shape)
    vectors = np.asarray([np.asarray(map[i][0]) for i in range(len(map))])

    clusters = kclusterer.cluster(vectors, assign_clusters=True)
    data_df_i = pd.DataFrame(list(
        zip([data["Reviews"][i] for i in indexes], [data["processedReviews"][i] for i in indexes],
            [data["Date"][i] for i in indexes], [data["Sentiments"][i] for i in indexes],
            [data["Lang"][i] for i in indexes], clusters)),
                             columns=['Reviews', 'processedReviews', 'Date', 'Sentiments', 'Lang', 'Cluster'])

    with open(outputfile_name, 'w+', encoding="utf-8") as file:
        data_df_i.to_csv(file, header=True, index=False, index_label=False)
        file.close()


def Plot_Clusters_Kmeans(outputfile_name, filename_data, filename_doc2vec, nbr_clusters, minimum_term_frequency, path):
    clustering(outputfile_name, filename_data, filename_doc2vec, nbr_clusters)
    list_stop_words = ["zoe", "vehicle", "electric", "volkswagen", "car", "ev", '’', '“', '”', "-", "--", "_"]
    list_stop_words = [stemmer.stem(list_stop_words[i]) for i in range(len(list_stop_words))]
    groups = []

    for key in range(int(nbr_clusters)):
        text2kw_clusters = pd.read_csv(outputfile_name, error_bad_lines=False, encoding='utf-8')
        m = text2kw_clusters[text2kw_clusters["Cluster"] == key]
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

        groups.append(str([c[i][0] for i in range(10)]))

    text2kw_clusters = pd.read_csv(outputfile_name, error_bad_lines=False, encoding='utf-8')
    text2kw_clusters['Cluster'] = text2kw_clusters['Cluster'].astype(str)
    corpus = (st.CorpusFromPandas(text2kw_clusters, category_col='Cluster', text_col='Reviews', nlp=nlp).build().\
              remove_terms(ENGLISH_STOP_WORDS, ignore_absences=True).
              get_unigram_corpus().compact(st.ClassPercentageCompactor(term_count=2,
                                               term_ranker=st.OncePerDocFrequencyRanker)))

    for i in range(nbr_clusters):
        html = st.produce_scattertext_explorer(corpus, category=str(i), category_name=str(i) + " Category",
                                               not_category_name='Other Categories',
                                               metadata=text2kw_clusters['Date'],
                                               minimum_term_frequency=int(text2kw_clusters.shape[0]*0.01))
        filename = path + str(i) + "_Category-VS-other categories.html"
        open(filename, 'wb+').write(html.encode('utf-8'))

    countries = ["english", "fr", "it", "no"]
    for country in countries:
        text2kw_clusters = pd.read_csv(outputfile_name, error_bad_lines=False, encoding='utf-8')
        text2kw_clusters['Cluster'] = text2kw_clusters['Cluster'].astype(str)
        text2kw_clusters['Lang'] = text2kw_clusters['Lang'].astype(str)
        text2kw_clusters = text2kw_clusters.loc[text2kw_clusters["Lang"] == str(country)]

        corpus = (st.CorpusFromPandas(text2kw_clusters, category_col='Cluster', text_col='Reviews', nlp=nlp).build().\
                  remove_terms(ENGLISH_STOP_WORDS, ignore_absences=True).
                  get_unigram_corpus().compact(st.ClassPercentageCompactor(term_count=2,
                                                   term_ranker=st.OncePerDocFrequencyRanker)))

        for i in range(nbr_clusters):
            html = st.produce_scattertext_explorer(corpus, category=str(i), category_name=str(i) + " Category",
                                                   not_category_name='Other Categories',
                                                   metadata=text2kw_clusters['Date'],
                                                   minimum_term_frequency=int(text2kw_clusters.shape[0]*0.01))
            filename = path + str(i) + "_" + str(country) +  "_Category-VS-other categories.html"
            open(filename, 'wb+').write(html.encode('utf-8'))

    for i in range(nbr_clusters):
        text2kw_clusters = pd.read_csv(outputfile_name , error_bad_lines=False, encoding='utf-8')
        text2kw_clusters['Cluster'] = text2kw_clusters['Cluster'].astype(str)
        text2kw_clusters['Sentiments'] = text2kw_clusters['Sentiments'].astype(str)
        text2kw_clusters['Date'] = text2kw_clusters['Date'].astype(str)
        text2kw_clusters = text2kw_clusters.loc[text2kw_clusters["Cluster"] == str(i)]
        text2kw_clusters = text2kw_clusters.loc[text2kw_clusters["Sentiments"] != "neutral"]
        corpus = (st.CorpusFromPandas(text2kw_clusters, category_col='Sentiments', text_col='Reviews', nlp=nlp).build().
                  remove_terms(ENGLISH_STOP_WORDS, ignore_absences=True))
        html = st.produce_scattertext_explorer(corpus, category="positive", category_name="Positive Verbatims",
                                               not_category_name='Negative Verbatims',
                                               metadata=text2kw_clusters['Date'],
                                               minimum_term_frequency=int(text2kw_clusters.shape[0]*0.01))
        filename = path + str(i) + "_Positive_Category-VS-Negative_Category.html"
        open(filename, 'wb+').write(html.encode('utf-8'))

    countries = ["english", "fr", "it", "no"]
    for i in range(nbr_clusters):
        for country in countries:
            text2kw_clusters = pd.read_csv(outputfile_name , error_bad_lines=False, encoding='utf-8')
            text2kw_clusters['Cluster'] = text2kw_clusters['Cluster'].astype(str)
            text2kw_clusters['Sentiments'] = text2kw_clusters['Sentiments'].astype(str)
            text2kw_clusters['Date'] = text2kw_clusters['Date'].astype(str)
            text2kw_clusters['Lang'] = text2kw_clusters['Lang'].astype(str)
            text2kw_clusters = text2kw_clusters.loc[text2kw_clusters["Lang"] == str(country)]
            text2kw_clusters = text2kw_clusters.loc[text2kw_clusters["Cluster"] == str(i)]
            text2kw_clusters = text2kw_clusters.loc[text2kw_clusters["Sentiments"] != "neutral"]
            corpus = (st.CorpusFromPandas(text2kw_clusters, category_col='Sentiments', text_col='Reviews', nlp=nlp).build().
                      remove_terms(ENGLISH_STOP_WORDS, ignore_absences=True))
            html = st.produce_scattertext_explorer(corpus, category="positive", category_name="Positive Verbatims",
                                                   not_category_name='Negative Verbatims',
                                                   metadata=text2kw_clusters['Date'],
                                                   minimum_term_frequency=int(text2kw_clusters.shape[0]*0.01))
            filename = path + str(i) + "_" + str(country) + "_Positive_Category-VS-Negative_Category.html"
            open(filename, 'wb+').write(html.encode('utf-8'))

def wordcloud_cluster_byIds_Kmeans(path, text2kw, nbr_clusters):
    list_stop_words = ["e-golf", "vehicle", "electric", "e", "golf", "volkswagen", "car", "wagen", "ev", "volks", "vw"]
    list_stop_words = [stemmer.stem(list_stop_words[i]) for i in range(len(list_stop_words))]
    data = pd.read_csv(text2kw, error_bad_lines=False, encoding='utf-8')
    nclust = nbr_clusters

    for i in range(nclust):
        data = pd.read_csv(text2kw, error_bad_lines=False, encoding='utf-8')
        liste = []
        data_i = data.loc[data["Cluster"] == i]
        try:
            liste = [word_tokenize(str(x)) for x in data_i["processedReviews"] if
                     not stemmer.stem(str(x)) in list_stop_words]

            words = ''
            for j in range(len(liste)):
                for k in range(len(liste[j])):
                    if liste[j][k] not in list_stop_words:
                        try:
                            words = words + liste[j][k] + ' '
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
            plt.savefig(str(path) + "/" + str(i) + "_WordCloud.png")
            plt.close()
        except:
            pass


def Plot_Time_Series_Kmeans(path, nbr_clusters, list_stop_words=["e-golf", "vehicle", "electric", "e", "golf", "volkswagen", "car", "wagen", "ev", "volks", "vw"]):
    list_stop_words = ["zoe", "vehicle", "electric", "e", "volkswagen", "car", "ev"]
    list_stop_words = [stemmer.stem(list_stop_words[i]) for i in range(len(list_stop_words))]
    all = pd.read_csv(str(path) + '/Reviews_Clusters.csv', encoding='utf8')
    colors_1 = [plt.cm.Blues, plt.cm.Reds, plt.cm.Greens, plt.cm.Purples, plt.cm.YlOrBr, plt.cm.Greys,
                plt.cm.YlGn, plt.cm.PuRd, plt.cm.bone, plt.cm.spring, plt.cm.afmhot, plt.cm.BuGn]

    dates = []
    for i in range(all.shape[0]):
        dates = []
        for i in range(all.shape[0]):
            try:
                if (isNaN(all["Date"][i]) == False) and (int(all["Date"][i].split("-")[0]) > int(2013)):
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
        plt.savefig(str(path) + "/Mentions-VS-Time.png")
        plt.close(fig)
from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer('english')
# nltk.download('vader_lexicon')
import spacy

spacy.prefer_gpu()
nlp = spacy.load('en_core_web_sm')
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
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
stopWords = stopWords+ENGLISH_STOP_WORDS+stopWordsGerman+stopWordsFrench
# nltk.download('vader_lexicon')

from gensim.models import Word2Vec

model = Word2Vec.load(r"C:\Users\p100623\PycharmProjects\WebScraping\Data_Processing\Word2vVec.model")

import spacy

nlp = spacy.load('en_core_web_sm')


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
    with open(output_filename, "a+", encoding="utf-8") as output:
        writer = csv.writer(output, lineterminator='\n')
        writer.writerows(doc2vecs)


def top_n_keywords_concat(label, word2vec_model, text2kw_words, text2kw_coef, vector_size=256):
    words = pd.read_csv(text2kw_words, header=None, error_bad_lines=False, encoding='utf-8')
    coefs = genfromtxt(text2kw_coef, delimiter=' ')
    words = words[0]
    labels = []
    X = np.zeros((len(words), 20, vector_size))  # len(words), dtype=K.floatx())
    for i in range(0, len(words)):  # len(words)):
        for j in range(0, len(words[i].split())):
            if words[i].split()[j] in word2vec_model:
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
    plt.savefig(path+"\ElbowCurve.png")
    plt.show(block=False)
    plt.close(fig)


def score_Review(Reviews):
    df_dict = {}
    for i, response in enumerate(Reviews):
        scorer = SentimentIntensityAnalyzer()
        scores = scorer.polarity_scores(str(response))
        df_dict[i] = scores

    df = pd.DataFrame.from_dict(df_dict, orient='columns')
    df = df.T

    diff_score = df['compound']
    for i in range(len(diff_score)):
        if diff_score[i] > 0.1:
            diff_score[i] = "positive"
        elif diff_score[i] < -0.1:
            diff_score[i] = "negative"
        else:
            diff_score[i] = "neutral"
    diff_score.columns = ["Sentiments"]
    sentiments = pd.concat([df, diff_score], axis=1)
    return sentiments


from Data_Processing.ReviewsSegmentation import *
import pandas as pd
import tensorflow as tf
import keras.backend as K
from keras.models import load_model


def f1(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp/(tp+fp+K.epsilon())
    r = tp/(tp+fn+K.epsilon())

    f1 = 2*p*r/(p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)


def f1_loss(y_true, y_pred):
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp/(tp+fp+K.epsilon())
    r = tp/(tp+fn+K.epsilon())

    f1 = 2*p*r/(p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return 1-K.mean(f1)


def sample(preds):
    lables_list = ["battery", "brakes", "dealer", "engine_gearbox", "equipements", "interior_seats", "price"]
    return lables_list[np.argmax(preds)]


def Cluster_key_words(data_path, path):
    data = pd.read_csv(data_path, error_bad_lines=False)
    word2vec_model = Word2Vec.load("Word2vVec.model")
    model_CONV1D = load_model('CONV1D.h5', custom_objects={'f1': f1, 'f1_loss': f1_loss})

    path = r"C:\Users\p100623\PycharmProjects\WebScraping\Data_Processing\Outputs\Flatten"
    X_test = top_n_keywords_concat_predict(word2vec_model, path+"\keywords.csv", path+"\coefs.csv", vector_size=256)

    prediction = model_CONV1D.predict(X_test)
    predictions_tab = []
    for t in range(len(prediction)):
        predictions_tab.append(sample(prediction[t]))

    data_custered = pd.DataFrame(list(zip(data["Car"], data["Text"], data["Date"], data["Country"], data["Sentiments"],
                                          data["processedReviews"], predictions_tab)),
                                 columns=['Car', 'Text', 'Date', 'Country', 'Sentiments', 'processedReviews',
                                          "Cluster"])

    directory = r"C:\Users\p100623\PycharmProjects\WebScraping\Data_Processing\Outputs\Classes\."
    try:
        os.mkdir(directory)
    except FileExistsError:
        pass
    try:
        os.remove(r"C:\Users\p100623\PycharmProjects\WebScraping\Data_Processing\Outputs\DataSets\Clustered_Data.csv")
    except:
        pass

    data_custered.to_csv(
        r"C:\Users\p100623\PycharmProjects\WebScraping\Data_Processing\Outputs\DataSets\Data_clustered.csv",
        header=True, index=False, index_label=False)


def Plot_Clusters_trained(filename_data, path, minimum_term_frequency):
    try:
        os.mkdir(path)
    except FileExistsError:
        shutil.rmtree(path)
        os.mkdir(path)

    text2kw_clusters = pd.read_csv(filename_data, error_bad_lines=False, encoding='utf-8')
    categories = list(set(text2kw_clusters["Cluster"]))
    categories = [cat for cat in categories if isNaN(cat) == False]
    categories.sort()

    nbr_clusters = len(categories)
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
    for group in groups:
        if count.get(group) is None:
            groups.remove(group)

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

    colors_2 = []
    colors_3 = []

    for i in range(nbr_clusters):
        colors_2.append(colors_1[i](0.9))
        colors_3.append(colors_1[i](0.8))
        colors_3.append(colors_1[i](0.6))
        colors_3.append(colors_1[i](0.4))

    # First Ring
    mypie, _ = ax.pie(group_size, radius=1.3, labels=group_names, labeldistance=1.05, colors=colors_2)
    plt.setp(mypie, width=0.3, edgecolor='white')

    # Second Ring (Inside)
    mypie2, _ = ax.pie(subgroup_size, radius=1.3-0.3, labels=subgroup_names, labeldistance=0.65, colors=colors_3,
                       rotatelabels=True, startangle=360, counterclock=True)
    plt.setp(mypie2, width=0.4, edgecolor='white')
    plt.margins(0, 0)

    plt.savefig(path[:-1]+"Sentiments.png", optimize=True)
    plt.rcParams['figure.figsize'] = 14, 7
    plt.close(fig)

    text2kw_clusters = pd.read_csv(filename_data, error_bad_lines=False, encoding='utf-8')
    text2kw_clusters['Cluster'] = text2kw_clusters['Cluster'].astype(str)
    corpus = (st.CorpusFromPandas(text2kw_clusters, category_col='Cluster', text_col='Text', nlp=nlp).build().
        remove_terms(stopWords, ignore_absences=True).
        get_unigram_corpus().compact(
        st.ClassPercentageCompactor(term_count=2, term_ranker=st.OncePerDocFrequencyRanker)))
    for i in categories:
        try:
            directory = path[:-1]+str(i)
            try:
                os.mkdir(directory)
            except FileExistsError:
                pass

            html = st.produce_scattertext_explorer(corpus, category=str(i), category_name=str(i)+" Category",
                                                   not_category_name='Other Categories',
                                                   metadata=text2kw_clusters['Date'],
                                                   minimum_term_frequency=int(text2kw_clusters.shape[0]*0.005))
            filename = directory+r"\\"+str(i)+"_Category-VS-other categories.html"
            open(filename, 'wb+').write(html.encode('utf-8'))
        except:
            pass

    text2kw_clusters = pd.read_csv(filename_data, error_bad_lines=False, encoding='utf-8')
    text2kw_clusters['Cluster'] = text2kw_clusters['Cluster'].astype(str)
    text2kw_clusters['Text'] = text2kw_clusters['Text'].apply(nlp)
    corpus = (st.CorpusFromParsedDocuments(text2kw_clusters, category_col='Cluster', parsed_col='Text')).build(). \
        remove_terms(stopWords, ignore_absences=True)
    for i in categories:
        try:
            directory = path[:-1]+str(i)
            try:
                os.mkdir(directory)
            except FileExistsError:
                pass

            html = word_similarity_explorer_gensim(corpus, category=str(i), category_name=str(i)+" Category",
                                                   not_category_name='Other Categories',
                                                   minimum_term_frequency=int(
                                                       text2kw_clusters.shape[0]*0.005),
                                                   target_term=stemmer.stem(str(i)),
                                                   # pmi_threshold_coefficient=4,
                                                   width_in_pixels=1000,
                                                   metadata=text2kw_clusters['Date'],
                                                   word2vec=model,
                                                   max_p_val=0.05,
                                                   save_svg_button=True)
            filename = directory+r"\\"+str(i)+"_w2v_Category-VS-other categories.html"
            open(filename, 'wb+').write(html.encode('utf-8'))
        except:
            pass

    for i, classe in enumerate(categories):
        try:
            directory = path[:-1]+str(classe)
            try:
                os.mkdir(directory)
            except FileExistsError:
                pass

            cat = ["battery", "brakes", "dealer", "interior", "engine", "equipements", "price"]
            text2kw_clusters = pd.read_csv(filename_data, error_bad_lines=False, encoding='utf-8')
            text2kw_clusters['Cluster'] = text2kw_clusters['Cluster'].astype(str)
            text2kw_clusters['Sentiments'] = text2kw_clusters['Sentiments'].astype(str)
            text2kw_clusters = text2kw_clusters.loc[text2kw_clusters["Cluster"] == str(classe)]
            text2kw_clusters = text2kw_clusters.loc[text2kw_clusters["Sentiments"] != "neutral"]
            text2kw_clusters['Text'] = text2kw_clusters['Text'].apply(nlp)
            corpus = (
                st.CorpusFromParsedDocuments(text2kw_clusters, category_col='Sentiments', parsed_col='Text')).build(). \
                remove_terms(stopWords, ignore_absences=True)
            html = word_similarity_explorer_gensim(corpus, category="positive", category_name="Positive Verbatims",
                                                   not_category_name='Negative Verbatims',
                                                   minimum_term_frequency=int(
                                                       text2kw_clusters.shape[0]*0.005),
                                                   target_term=stemmer.stem(str(cat[i])),
                                                   # pmi_threshold_coefficient=4,
                                                   width_in_pixels=1000,
                                                   metadata=text2kw_clusters['Date'],
                                                   word2vec=model,
                                                   max_p_val=0.05,
                                                   save_svg_button=True)
            filename = directory+r"\\"+str(classe)+"_w2v__Positive_Category-VS-Negative_Category.html"
            open(filename, 'wb+').write(html.encode('utf-8'))
        except:
            pass

    for i in categories:
        try:
            directory = path[:-1]+str(i)
            try:
                os.mkdir(directory)
            except FileExistsError:
                pass

            text2kw_clusters = pd.read_csv(filename_data, error_bad_lines=False, encoding='utf-8')
            text2kw_clusters['Cluster'] = text2kw_clusters['Cluster'].astype(str)
            text2kw_clusters['Sentiments'] = text2kw_clusters['Sentiments'].astype(str)
            text2kw_clusters = text2kw_clusters.loc[text2kw_clusters["Cluster"] == str(i)]
            text2kw_clusters = text2kw_clusters.loc[text2kw_clusters["Sentiments"] != "neutral"]
            corpus = st.CorpusFromPandas(text2kw_clusters, category_col='Sentiments', text_col='Text', nlp=nlp).build(). \
                remove_terms(stopWords, ignore_absences=True)
            html = st.produce_scattertext_explorer(corpus, category="positive", category_name="Positive Verbatims",
                                                   not_category_name='Negative Verbatims',
                                                   metadata=text2kw_clusters['Date'],
                                                   minimum_term_frequency=int(cluster.shape[0]*0.005))
            filename = directory+r"\\"+str(i)+"_Positive_Category-VS-Negative_Category.html"
            open(filename, 'wb+').write(html.encode('utf-8'))
        except:
            pass


def wordcloud_cluster_byIds(filename, path):
    data = pd.read_csv(filename, encoding='utf8')
    categories = list(set(data["Cluster"]))
    categories = [cat for cat in categories if isNaN(cat) == False]
    categories.sort()

    list_stop_words = []
    list_stop_words = [stemmer.stem(list_stop_words[i]) for i in range(len(list_stop_words))]

    for i, key in enumerate(categories):
        data = pd.read_csv(filename, error_bad_lines=False, encoding='utf-8')
        data_i = data.loc[data["Cluster"] == key]

        try:
            liste = [word_tokenize(str(x)) for x in data_i["processedReviews"] if
                     not stemmer.stem(str(x)) in list_stop_words]
            words = ''
            for j in range(len(liste)):
                for k in range(len(liste[j])):
                    if liste[j][k] not in list_stop_words and liste[j][k]:
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

            # plt.rcParams['figure.figsize'] = 8, 8
            plt.tight_layout()
            plt.imshow(wordcloud)
            plt.axis("off")
            plotname = path+r"\\Outputs\\Classes\\"+str(key)+r"\\WordCloud.png"
            plt.savefig(plotname)
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
                        if liste[j][k] not in list_stop_words and liste[j][k]:
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

                plt.rcParams['figure.figsize'] = 13, 10
                plt.tight_layout()
                plt.imshow(wordcloud, interpolation="bilinear")
                plt.axis("off")
                plotname = path+r"\\Outputs\\Classes\\"+str(key)+r"\\"+str(sent).upper()+"_WordCloud.png"
                plt.savefig(plotname)
                plt.close()
            except:
                pass


def Plot_Time_Series(filename, path):
    all = pd.read_csv(filename, encoding='utf8')

    categories = list(set(all["Cluster"]))
    categories = [cat for cat in categories if isNaN(cat) == False]
    categories.sort()

    colors_1 = [plt.cm.Blues, plt.cm.Reds, plt.cm.Greens, plt.cm.Purples, plt.cm.Greys,
                plt.cm.YlGn, plt.cm.PuRd, plt.cm.bone, plt.cm.spring, plt.cm.afmhot, plt.cm.BuGn]

    all = pd.read_csv(filename, encoding='utf8')

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
    for i, key in enumerate(categories):
        m = dates_data[dates_data["Cluster"] == key].resample("15d").count()
        with plt.style.context('bmh'):
            # plot command goes here
            try:
                m["Date"] = list(m.index)
                m.plot(x="Date", y='Cluster', ax=ax1, lw=line_width, label=str(key), color=colors_1[i](0.8),
                       figsize=(12, 8))
            except:
                pass

    ax1.set_xlabel('Date')
    ax1.set_ylabel('Number of Mentions', color='black')
    with plt.style.context('bmh'):
        plt.rcParams['figure.figsize'] = 7, 5
        plt.savefig(path+r"\Outputs\Classes\Mentions-VS-Time.png")
        plt.close(fig)

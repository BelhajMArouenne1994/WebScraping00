from Data_Processing.CleanDataSets import *
from Data_Processing.brandy import *
from Data_Processing.reviews_segmentation import *
from Data_Processing.CleanFacebook import *
import pandas as pd
from gensim.models import Word2Vec
os.chdir(r"/home/marwen/PycharmProjects/Renault/CARS/ZOE/ZOE (Forums) (anglais)")
###-------------------------------------------------###


"""
###----------------------------------- Downloading mentions from Brandwatch API -------------------------------------###
user_name = "marouenne.belhaj@renault.com"
password = "marwen1A?"
access_token = get_new_key(user_name, password)
project_list = boot_brandy(access_token)
project_name = "E-GOLF"
project_id = get_project_id_from_name(project_list, project_name)
query_list = get_query_id(project_id, access_token)
query_id = "1999806841"
stream_data("DataSets/E-GOLF-anglais_brandWatch.csv", '2012-01-01', "2019-07-01", project_id, query_id, access_token, new=True)
#clean_brandwatch("intermediate/E-GOLF_clean.csv", 'DataSets/E-GOLF-anglais_brandWatch.csv', "en", "", add=False, classe=False)

clean_posts_database("intermediate/ZE_Owners_Club_fb_posts_clean.csv",
                     "DataSets/ZE_Owners_Club_fb_posts.csv", add=False, language="english")

clean_posts_database("intermediate/ZE_Owners_Club_fb_comments_clean.csv",
                     "DataSets/ZE_Owners_Club_fb_comments.csv", add=False, language="english")

clean_posts_database("intermediate/Renault_ZOE_Francophone_fb_posts_clean.csv",
                     "DataSets/Renault_ZOE_Francophone_FB_posts.csv", add=False, language="fr")

clean_posts_database("intermediate/Renault_ZOE_Francophone_fb_comments_clean.csv",
                     "DataSets/‎Renault_ZOE_Francophone_FB_comments.csv", add=False, language="fr")

clean_posts_database("intermediate/Renault_Zoe-Italia_e_Svizzera_fb_posts_clean.csv",
                     "DataSets/Renault_Zoe-Italia_e_Svizzera_fb_posts.csv", add=False, language="it")

clean_posts_database("intermediate/Renault_Zoe-Italia_e_Svizzera_fb_comments_clean.csv",
                     "DataSets/Renault_Zoe-Italia_e_Svizzera_fb_comments.csv", add=False, language="it")
                     
clean_forum_database("intermediate/zoe_forum_clean.csv", "DataSets/zoe_forum.csv", add=False, language="fr")
###------------------------------------------------------------------------------------------------------------------###

clean_forum_database_chunck("intermediate/zoe_goingelectric_clean", "DataSets/zoe_goingelectric.csv", add=False, language="de")


clean_forum_database("intermediate/‎zoe_elbilforum_clean.csv", "DataSets/‎zoe_elbilforum.csv", add=False, language="no")

clean_forum_database("intermediate/zoe_forum_voiture_propre_clean.csv", "DataSets/zoe_forum_voiture_propre.csv", add=False, language="fr")
"""
###--------------- ----Concating Data Sets and removing verbatims that do not concern ZOE model----------------------###
links = []

others = ["yamaha", "lexus", "tesla", "lotus", "volvo", "porsche", "mercedes", "benz", "outlander",
          "mitsubishi", "captur", "clio", "megane", "twizy", "kangoo", "audi", "e-tron", "bmw", "i3",
          "chevrolet", "chevy", "bolt", "fiat", "500e", "honda", "clarity", "Ioniq", "hyundai", "kona",
          "jaguar", "i-pace", "niro", "kia", "soul", "e-golf", "nissan", "leaf", "tesla", "mahindra",
          "toyota", "jetta", "gti", "weekend", "model-s"]
others = [stemmer.stem(x.lower()) for x in others]

paths = ["intermediate/‎zoe_elbilforum_clean.csv", "intermediate/zoe_forum_clean.csv",
         "intermediate/ZE_Owners_Club_fb_posts_clean.csv", "intermediate/ZE_Owners_Club_fb_comments_clean.csv",
         "intermediate/Renault_Zoe-Italia_e_Svizzera_fb_posts_clean.csv",
         "intermediate/Renault_Zoe-Italia_e_Svizzera_fb_comments_clean.csv",
         "intermediate/Renault_ZOE_Francophone_fb_posts_clean.csv",
         "intermediate/Renault_ZOE_Francophone_fb_comments_clean.csv"]
output = "intermediate/Reviews.csv"


with open("intermediate/Reviews.csv", 'a+') as f:
    f.write("ReviewsµprocessedReviewsµDateµLang")
    f.write("\n")
f.close()

def concatReviews(path, output):
    forum = pd.read_csv(path, error_bad_lines=False, encoding='utf-8',
                        delimiter=",", lineterminator='\n')
    for i in range(forum.shape[0]):
        if sum([1 for x in others if str(forum["Text"][i]).lower().find(x) != -1]) < 1 and \
                sum([1 for x in links if str(forum["Text"][i]).lower().find(x) != -1]) < 1:
            if forum["processedReviews"][i] == forum["processedReviews"][i]:
                with open(output, 'a+') as f:
                    f.write(str(forum["Text"][i].lower().replace("\n", " ").replace('"', " ").replace("'", " ").replace(",", ".").
                                replace('r-link', "rlink")).replace("r link", "rlink").replace("r_link", "rlink").
                            replace('my renault', "myrenault").replace("my_renault", "myrenault").replace("my-renault", "myrenault"))
                    f.write("µ")
                    f.write(str(forum["processedReviews"][i]))
                    f.write("µ")
                    f.write(forum["Date"][i])
                    f.write("µ")
                    f.write(forum["language"][i])
                    f.write("\n")
                f.close()
    del forum

for file in paths:
    concatReviews(file, output)
###---------------------------------------------------------------------------###


###----------------------------Removing Duplicates----------------------------###
processedReviews = pd.read_csv("intermediate/Reviews.csv", delimiter="µ", engine='python')

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
with open("intermediate/processedReviews.csv", 'w+', encoding="utf-8") as file:
    processedReviews_clean.to_csv(file, header=True, encoding="utf-8", index=False, index_label=False)
    file.close()
del processedReviews
del processedReviews_clean
###-------------------------------###


###--------------------------------------------Sentiment Analysis----------------------------------------------------###
test = pd.read_csv("intermediate/processedReviews.csv", error_bad_lines=False)
diff_score = score_Review(test["Reviews"])
sentiments = pd.concat([test, diff_score], axis=1)
sentiments.columns = ['Reviews', 'processedReviews', 'Date', "Lang", "compound", "neg", "neu", "pos", "Sentiments"]
drop_lines = []
for i in range(sentiments.shape[0]):
    if int(sentiments["neu"][i]) > 0.8:
        drop_lines.append(i)
sentiments.index = range(len(sentiments.index))
sentiments = sentiments.drop(drop_lines, axis=0)
sentiments.index = range(len(sentiments.index))
del test
del diff_score
with open("intermediate/Reviews_final.csv", 'w+', encoding="utf-8") as file:
    sentiments.to_csv(file, header=True, encoding="utf-8", index=False, index_label=False)
    file.close()


sent = pd.read_csv("intermediate/Reviews_final.csv", error_bad_lines=False)
sent['Sentiments'] = sent['Sentiments'].astype(str)
negatives = sent.loc[sent["Sentiments"] == "negative"]
negatives.index = range(len(negatives.index))
with open("intermediate/negatives.csv", 'w+', encoding="utf-8") as file:
    negatives.to_csv(file, header=True, index=False, index_label=False)
    file.close()
###------------------------------------------------------------------------------------------------------------------###

model = Word2Vec.load("DE_w2v.model")
data_1 = pd.read_csv("intermediate/Reviews_final_DE.csv", error_bad_lines=False)
data_2 = pd.read_csv("intermediate/Reviews_final.csv", error_bad_lines=False)
frames = [data_1, data_2]
data = pd.concat(frames)
data.to_csv("all.csv", header=True, index=False, index_label=False)

data.Lang.value_counts()

###--- Clustering Data ---###
"""
    Clustering Data and Deleting those which are not related to our Topic
"""
#model = Word2Vec.load("Word2Vec/models/EVs_word2vec (10 iters).model")

###-----------------------##
def flatten(directory, processedReviews_file, max_df):
    try:
        # Create target Directory
        os.mkdir(directory)
        print("Directory ", directory, " Created ")
    except FileExistsError:
        print("Directory ", directory, " already exists")
    text2kw_keywords = extract_tfidf_keywords(directory + "/keywords.csv", processedReviews_file, 20, max_df)
    text2kw_coefs = extract_tfidf_coefs(directory + "/coefs.csv", processedReviews_file, 20, max_df)
    top_n_keywords_average(directory + "/text2kw_top_n_keywords.csv", model, directory + "/keywords.csv", directory + "/coefs.csv",
                           vector_size=256)
    ElbowMethod(directory + "/", directory + "/text2kw_top_n_keywords.csv", 15)
###-----------------------###

path = "K_means_1"
flatten(path, "all.csv", max_df=0.7)

###------------------Scatter Plots------------------###
for i in [6, 7, 8, 9]:
    path = "K_means_1"
    path_plots = path + "/Clusters" + str(i)
    try:
        # Create target Directory
        os.mkdir(path_plots)
        print("Directory ", path_plots, " Created ")
    except FileExistsError:
        print("Directory ", path_plots, " already exists")

    Plot_Clusters_Kmeans(path_plots + "/Reviews_Clusters.csv", "all.csv",
                  path + "/text2kw_top_n_keywords.csv", i, 10, path_plots + "/")

    wordcloud_cluster_byIds_Kmeans(path_plots,  path_plots +"/Reviews_Clusters.csv", i)
    Plot_Time_Series_Kmeans(path_plots, i)

    for j in range(i):
        path = "Kmeans1/Clusters" + str(i) + "/Clusters" + str(j)
        try:
            # Create target Directory
            os.mkdir(path)
            print("Directory ", path, " Created ")
        except FileExistsError:
            print("Directory ", path, " already exists")

        clusters = pd.read_csv(path_plots + "/Reviews_Clusters.csv", error_bad_lines=False, encoding='utf-8')
        clusters['Cluster'] = clusters['Cluster'].astype(str)
        cluster = clusters.loc[clusters["Cluster"] == str(j)]
        cluster.index = range(len(cluster.index))
        cluster.to_csv(path + "/processedReviews2.csv", header=True, index=False, index_label=False)
        del cluster
        flatten(path, path + "/processedReviews2.csv", max_df=0.7)

        ###------------------Scatter Plots------------------###

        for k in [2, 3, 4, 5, 6, 7, 8]:
            path_2 = "Clusters" + str(i) + "/Clusters" + str(j) + "/" + str(k)
            try:
                # Create target Directory
                os.mkdir(path_2)
                print("Directory ", path_2, " Created ")
            except FileExistsError:
                print("Directory ", path_2, " already exists")

            Plot_Clusters_Kmeans(path_2 + "/Reviews_Clusters.csv", path + "/processedReviews2.csv",
                          path + "/text2kw_top_n_keywords.csv", k, 4, path_2 + "/")
            plt.close()

            wordcloud_cluster_byIds(path_2, path_2 + "/Reviews_Clusters.csv", k)
            plt.close()
            Plot_Time_Series(path_2, k)
###-------------------------------------------------###

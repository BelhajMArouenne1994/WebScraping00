from Data_Processing.CleanDataSets import *
from Data_Processing.brandy import *
from Data_Processing.reviews_segmentation import *
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


categories = ["battery", "brakes", "dealer", "doors-interior-seats", "engine-gearbox", "equipements", "price"]

model = Word2Vec.load("DE_w2v.model")

print("Positive")
similar_words = {search_term: [item[0] for item in model.wv.most_similar(positive=[search_term], topn=10)]
                 for search_term in [stemmer.stem("autonomy")]}
pprint.pprint(similar_words, compact=True)



battery = ['pack', 'nimh', 'lithium-ion', 'recharg', 'cell', 'lead-acid', 'kwh', 'li-ion', '12v', 'overcharg',
           'ultracapacitor', 'discharg', 'cathod', 'storag', 'recharg', 'charger', 'fast-charg', 'refuel',
           'discharg', 'soc', 'overnight', 'plug', 'commut', 'deplet', 'wallbox', 'dock', 'replenish',
           'supercharg', 'socket', 'outlet', 'cabl', 'connector', 'hook', 'plug-in', 'recharg', 'cord',
           'volt', 'charg', 'batteri', 'charg', 'km', 'mile']

brakes = ['deceler', 'hydraul', 'clutch', 'ebd', 'suspens', 'assist', 'pedal', 'regener', 'acceler', 'transmiss',
          'steer', 'drum', 'flywheel', 'tpms', 'mechan', 'throttl', 'wheel', 'ab', 'airbag', 'aeb', 'pressur',
          'damper', 'vsc', 'regen', 'axl', 'start-stop', 'downhil', 'valet', 'stop-start', 'calip', 'pump', 'kinet',
          'brake', 'coast']

engine_gearbox = ['turbocharg', 'liter', '20-liter', 'four-cylind', 'cylind', 'twin-turbo', 'v6', 'turbo', '4-cylind',
                  'crankshaft', '20-litr', 'motor', 'powertrain', 'v-8', '25-litr', 'motorgener', 'v8', '4cyl',
                  'powerpl',
                  'inlin', 'v-6', '20l', '16-litr', 'hp', 'motor-gener', 'transmiss', 'seven-spe', 'six-spe', 'tranni',
                  'tran', '6-speed', 'eight-spe', 'clutch', 'dual-clutch', 'cvt', '7-speed', '5-speed', '8-speed',
                  'transaxl', 'mate', 'dsg', 'gear', 'shifter', 'flywheel', 'turbo', 'drivelin', 'amt', 'paddl', 'v6',
                  'engin', 'gearbox']

equipements = ['instal', 'applianc', 'featur', 'portabl', 'satellit', 'function', 'option', 'driver-assist',
               'upgrad', 'safeti', 'conveni', 'infotain', 'enhanc', 'ipod', 'usb', 'aux', 'sync', 'audio',
               'hands-fre', 'sirius', 'radio', 'compat', 'mp3', 'lamp', 'headlamp', 'illumin', 'heavi', 'bulb',
               'light-emit', 'headlight', 'bright', 'hid', 'laser', 'radar', 'infrar', 'mirror', 'monitor',
               'lidar', 'gps', 'navig', 'hud', 'telephon', 'smartphon', 'iphon', 'laptop', 'bluetooth',
               'app', 'notebook', 'email', 'tablet', 'camera', 'light', 'equip', 'rlink', 'phone']

interior_seats = []
interior_seats.append([item[0] for item in model.wv.most_similar(positive=[stemmer.stem("doors")], topn=16)])
interior_seats.append([item[0] for item in model.wv.most_similar(positive=[stemmer.stem("interior")], topn=16)])
interior_seats.append([item[0] for item in model.wv.most_similar(positive=[stemmer.stem("seats")], topn=30)])
interior_seats = list(itertools.chain(*interior_seats))
interior_seats.append(stemmer.stem("doors"))
interior_seats.append(stemmer.stem("interior"))
interior_seats.append(stemmer.stem("seats"))
interior_seats.append(stemmer.stem("belt"))
interior_seats.append(stemmer.stem("flap"))
interior_seats.append(stemmer.stem("windsheld"))


price = ['msrp', 'cost', 'inventori', 'nada', 'discount', 'tax', 'k', 'cheapest', 'cheap', 'resal', 'expens',
         'fee', 'yuan', 'expens', 'price', 'cheaper', "money", 'save', 'milag', 'mileag', 'yield', 'incent',
         'rebat', 'exempt', 'fee', 'excis', 'subsidi', 'deduct', 'taxat', 'gst', 'pay', 'credit', 'incom',
         'bill', 'price', 'euro', 'pound', 'dollar', '£']

dealer = ['dealership', 'shop', 'inventori', 'franchis', 'retail', 'custom', 'pre-own', 'repair', 'servic',
          'sell', 'showroom', 'buyer', 'technician', 'distributor', 'purchas', 'invoic', 'inspect', 'salesman',
          'leas', 'financ', 'rental', 'loan', 'purchas', 'payment', 'credit', 'ownership', 'rent', 'lender',
          'scrap', 'tax', 'rebat', 'insur', 'fee', 'trade-in', 'approv', 'cash', 'payabl', 'paid', 'dealer']

###-------------------------------------------------###
data = pd.read_csv("all.csv", error_bad_lines=False)
from collections import Counter
Counter(data["Lang"])

categories = ["battery", "brakes", "dealer", "doors-interior-seats", "engine-gearbox", "equipements", "price"]
list_stop_words = ["zoe", "renault", "car", "vehicle", "vw", "electric-car"]

preds = []
for i in range(data.shape[0]):
    words = str(data["processedReviews"][i]).split()
    counts = []
    results_battery = 0
    results_brakes = 0
    results_engine_gearbox = 0
    results_equipements = 0
    results_interior_seats = 0
    results_price = 0
    results_dealer = 0
    for w in words:
        if w not in list_stop_words:
            results_battery += battery.count(w)
            results_brakes += brakes.count(w)
            results_engine_gearbox += engine_gearbox.count(w)
            results_equipements += equipements.count(w)
            results_interior_seats += interior_seats.count(w)
            results_price += price.count(w)
            results_dealer += dealer.count(w)

    results_battery = results_battery / len(battery)
    results_brakes = results_brakes / len(brakes)
    results_engine_gearbox = results_engine_gearbox / len(engine_gearbox)
    results_equipements = results_equipements / len(equipements)
    results_interior_seats = results_interior_seats / len(interior_seats)
    results_price = results_price / len(price)
    results_dealer = results_dealer / len(dealer)

    counts = [results_battery, results_brakes, results_dealer, results_interior_seats,
              results_engine_gearbox, results_equipements, results_price]

    if max(counts) > 0:
        categorie = categories[np.argmax(counts)]
        preds.append(categorie)
    else:
        preds.append(None)

data_custered = pd.DataFrame(list(zip(data["Reviews"], data["processedReviews"],
                                      data["Date"], data["Sentiments"], data["Lang"], preds)),
                             columns=['Reviews', 'processedReviews', 'Date', 'Sentiments', 'Lang', 'Cluster'])

data_custered.to_csv("ZOE_clustered.csv", header=True, index=False, index_label=False)

########################################################################################################################

###------------------------------------------------------------------------------------------------------------------###
sent = pd.read_csv("ZOE_clustered.csv", error_bad_lines=False)
sent['Sentiments'] = sent['Sentiments'].astype(str)
negatives = sent.loc[sent["Sentiments"] == "negative"]
negatives.index = range(len(negatives.index))
with open("ZOE_clustered_negatives.csv", 'w+', encoding="utf-8") as file:
    negatives.to_csv(file, header=True, index=False, index_label=False)
    file.close()
###------------------------------------------------------------------------------------------------------------------###

Plot_Clusters_by_country("ZOE_clustered.csv", "")
Plot_Clusters_trained("ZOE_clustered.csv", "")
Plot_Time_Series("ZOE_clustered.csv")
wordcloud_cluster_byIds("ZOE_clustered.csv")
#############################################


###--- Clustering Data ---###
"""
    Clustering Data and Deleting those which are not related to our Topic
"""
directory = "Kmeans_2"
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
                           vector_size=256)
    #ElbowMethod(directory + "/", directory + "/text2kw_top_n_keywords.csv", 15)
###-----------------------###
sent = pd.read_csv("ZOE_clustered.csv", error_bad_lines=False)
sent['Sentiments'] = sent['Sentiments'].astype(str)
for categorie in categories:
    path = "Kmeans_2/" + categorie
    cat = pd.read_csv("ZOE_clustered.csv", error_bad_lines=False)
    cat['Cluster'] = cat['Cluster'].astype(str)
    cat = cat.loc[sent["Cluster"] == str(categorie)]
    cat.index = range(len(cat.index))
    with open(str(categorie) + ".csv", 'w+', encoding="utf-8") as file:
        cat.to_csv(file, header=True, index=False, index_label=False)
        file.close()
    flatten(path, str(categorie) + ".csv", max_df=0.8)
    for i in [8, 9, 10, 11]:
        path_plots = "Kmeans_2/" + categorie + "/Clusters_" + str(categorie) + "_" + str(i)
        try:
            # Create target Directory
            os.mkdir(path_plots)
            print("Directory ", path_plots, " Created ")
        except FileExistsError:
            print("Directory ", path_plots, " already exists")

        Plot_Clusters_Kmeans(path_plots + "/Reviews_Clusters.csv", str(categorie) + ".csv",
                             path + "/text2kw_top_n_keywords.csv", i, 1, path_plots + "/")

        wordcloud_cluster_byIds_Kmeans(path_plots, path_plots + "/Reviews_Clusters.csv", i)
        #Plot_Time_Series_Kmeans(path_plots, i)

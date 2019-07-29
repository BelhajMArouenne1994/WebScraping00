from Data_Processing.CleanDataSets import *
from Data_Processing.brandy import *
from Data_Processing.reviews_segmentation import *
import numpy as np
import pandas as pd
import multiprocessing
import datetime
import gensim
from gensim.models import Word2Vec
from os import listdir
from os.path import isfile, join
import itertools
import random
import pprint

os.chdir(r"/home/marwen/PycharmProjects/Renault/CARS/E-GOLF/ZOE (Forums) (anglais)/Word2Vec")
###-------------------------------------------------###


###----------------------------------- Downloading mentions from Brandwatch API -------------------------------------###
user_name = "marouenne.belhaj@renault.com"
password = "marwen1A?"
access_token = get_new_key(user_name, password)
project_list = boot_brandy(access_token)
start_date = "2014-01-01"

"""
###----------------------------------------------------------------------------------------------------------###
project_name = "EVs"
project_id = get_project_id_from_name(project_list, project_name)
query_list = get_query_id(project_id, access_token)
query_id = "1999778052"
stream_data("TrainingDataSets/EVs_brandWatch.csv", start_date, str(datetime.datetime.today().date()), project_id, query_id, access_token, new=True)
query_id = "1999777723"
stream_data("TrainingDataSets/EVs_brandWatch.csv", start_date, str(datetime.datetime.today().date()), project_id, query_id, access_token, new=False)
query_id = "1999772246"
stream_data("TrainingDataSets/EVs_brandWatch.csv", start_date, str(datetime.datetime.today().date()), project_id, query_id, access_token, new=False)
clean_brandwatch("TrainingDataSets/EVs_clean.csv", "TrainingDataSets/EVs_brandWatch.csv", "en", "", add=False, classe=False)

project_name = "HEV"
project_id = get_project_id_from_name(project_list, project_name)
query_list = get_query_id(project_id, access_token)
query_id = "1999796744"
stream_data("TrainingDataSets/HEVs_brandWatch.csv", start_date, str(datetime.datetime.today().date()), project_id, query_id, access_token, new=True)
clean_brandwatch("TrainingDataSets/HEVs_clean.csv", "TrainingDataSets/HEVs_brandWatch.csv", "en", "", add=False, classe=False)

project_name = "PHEV"
project_id = get_project_id_from_name(project_list, project_name)
query_list = get_query_id(project_id, access_token)
query_id = "1999796739"
stream_data("TrainingDataSets/PHEVs_brandWatch.csv", start_date, str(datetime.datetime.today().date()), project_id, query_id, access_token, new=True)
clean_brandwatch("TrainingDataSets/PHEVs_clean.csv", "TrainingDataSets/PHEVs_brandWatch.csv", "en", "", add=False, classe=False)
###----------------------------------------------------------------------------------------------------------###
"""

###------------------------------------------------------------------------------------------------------------------###
project_name = "Training data"
project_id = get_project_id_from_name(project_list, project_name)
query_list = get_query_id(project_id, access_token)

query_id = "1999810080"
stream_data("TrainingDataSets/equipements.csv", start_date, "2019-07-01", project_id, query_id, access_token, new=True)
clean_brandwatch("TrainingDataSets/equipements_clean.csv", 'TrainingDataSets/equipements.csv', "en", "", add=False, classe=False)
###------------------------------------------------------------------------------------------------------------------###
query_id = "1999810056"
stream_data("TrainingDataSets/interior-seats.csv", start_date, "2019-07-01", project_id, query_id, access_token, new=True)
clean_brandwatch("TrainingDataSets/interior-seats_clean.csv", 'TrainingDataSets/interior-seats.csv', "en", "", add=False, classe=False)
###------------------------------------------------------------------------------------------------------------------###
query_id = "1999810070"
stream_data("TrainingDataSets/brakes.csv", start_date, "2019-07-01", project_id, query_id, access_token, new=True)
clean_brandwatch("TrainingDataSets/brakes_clean.csv", 'TrainingDataSets/brakes.csv', "en", "", add=False, classe=False)
###------------------------------------------------------------------------------------------------------------------###
query_id = "1999810026"
stream_data("TrainingDataSets/lights.csv", start_date, "2019-07-01", project_id, query_id, access_token, new=True)
clean_brandwatch("TrainingDataSets/lights_clean.csv", 'TrainingDataSets/lights.csv', "en", "", add=False, classe=False)
###------------------------------------------------------------------------------------------------------------------###
query_id = "1999810002"
stream_data("TrainingDataSets/doors.csv", start_date, "2019-07-01", project_id, query_id, access_token, new=True)
clean_brandwatch("TrainingDataSets/doors_clean.csv", 'TrainingDataSets/doors.csv', "en", "", add=False, classe=False)
###------------------------------------------------------------------------------------------------------------------###
query_id = "1999809900"
stream_data("TrainingDataSets/engine-gearbox.csv", start_date, "2019-07-01", project_id, query_id, access_token, new=True)
clean_brandwatch("TrainingDataSets/engine-gearbox_clean.csv", 'TrainingDataSets/engine-gearbox.csv', "en", "", add=False, classe=False)
###------------------------------------------------------------------------------------------------------------------###
query_id = "1999810070"
stream_data("TrainingDataSets/battery.csv", start_date, "2019-07-01", project_id, query_id, access_token, new=True)
clean_brandwatch("TrainingDataSets/battery_clean.csv", 'TrainingDataSets/battery.csv', "en", "", add=False, classe=False)
###------------------------------------------------------------------------------------------------------------------###
query_id = "1999820928"
stream_data("TrainingDataSets/price.csv", "2014-01-01", "2019-07-01", project_id, query_id, access_token, new=True)
clean_brandwatch("TrainingDataSets/price_clean.csv", 'TrainingDataSets/price.csv', "en", "", add=False, classe=False)
###------------------------------------------------------------------------------------------------------------------###





###------------------------------------------------------------------------------------------------------------------###
mypath = "/home/marwen/PycharmProjects/Renault/CARS/E-GOLF/ZOE (Forums) (anglais)/Word2Vec/TrainingDataSets"
onlyfiles = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f)) and f.find("clean") != -1]
class corpus(object):
    def __iter__(self):
        for file in onlyfiles:
            for line in pd.read_csv(file, delimiter=',', error_bad_lines=False)["processedReviews"]:
                yield str(line).lower().split()

#word2vec_model = gensim.models.Word2Vec(corpus(), size=128, window=5, min_count=1000, workers=4, iter=50)
#word2vec_model.save("/home/marwen/PycharmProjects/Renault/CARS/E-GOLF/ZOE (Forums) (anglais)/Word2Vec/models/EVs_word2vec (10 iters).model")

word2vec_model = Word2Vec.load("/home/marwen/PycharmProjects/Renault/CARS/E-GOLF/ZOE (Forums) (anglais)/Word2Vec/models/EVs_word2vec (10 iters).model")
###------------------------------------------------------------------------------------------------------------------###


###------------------------------------------------------------------------------------------------------------------###
def flatten(directory, processedReviews_file, max_df, min_df):
    model = word2vec_model
    try:
        # Create target Directory
        os.mkdir(directory)
        print("Directory ", directory, " Created")
    except FileExistsError:
        print("Directory ", directory, " already exists")
    text2kw_keywords = extract_tfidf_keywords(directory + "/keywords.csv", processedReviews_file, 20, max_df, min_df)
    text2kw_coefs = extract_tfidf_coefs(directory + "/coefs.csv", processedReviews_file, 20, max_df, min_df)
###-----------------------###


###------------------------------------------------------------------------------------------------------------------###
path = "TrainingDataSets/equipements"
flatten(path, "TrainingDataSets/equipements_clean.csv", max_df=1.0, min_df=0.05)
X_equipements, labels_equipements = top_n_keywords_concat("equipements", word2vec_model, path + "/keywords.csv",
                                                          path + "/coefs.csv", vector_size=128)

path = "TrainingDataSets/interior-seats"
flatten(path, "TrainingDataSets/interior-seats_clean.csv", max_df=1.0, min_df=0.05)
X_interior_seats, labels_interior_seats = top_n_keywords_concat("interior-seats", word2vec_model, path + "/keywords.csv",
                                                          path + "/coefs.csv", vector_size=128)

path = "TrainingDataSets/brakes"
flatten(path, "TrainingDataSets/brakes_clean.csv", max_df=1.0, min_df=0.05)
X_brakes, labels_brakes = top_n_keywords_concat("brakes", word2vec_model, path + "/keywords.csv",
                                                path + "/coefs.csv", vector_size=128)

path = "TrainingDataSets/lights"
flatten(path, "TrainingDataSets/lights_clean.csv", max_df=1.0, min_df=0.05)
X_lights, labels_lights = top_n_keywords_concat("lights", word2vec_model, path + "/keywords.csv",
                                                          path + "/coefs.csv", vector_size=128)

path = "TrainingDataSets/doors"
flatten(path, "TrainingDataSets/doors_clean.csv", max_df=1.0, min_df=0.05)
X_doors, labels_doors = top_n_keywords_concat("doors", word2vec_model, path + "/keywords.csv",
                                                          path + "/coefs.csv", vector_size=128)

path = "TrainingDataSets/engine-gearbox"
flatten(path, "TrainingDataSets/engine-gearbox_clean.csv", max_df=1.0, min_df=0.05)
X_engine_gearbox, labels_engine_gearbox = top_n_keywords_concat("engine-gearbox", word2vec_model, path + "/keywords.csv",
                                                          path + "/coefs.csv", vector_size=128)

path = "TrainingDataSets/battery"
flatten(path, "TrainingDataSets/battery_clean.csv", max_df=1.0, min_df=0.05)
X_battery, labels_battery = top_n_keywords_concat("battery", word2vec_model, path + "/keywords.csv",
                                                          path + "/coefs.csv", vector_size=128)
###--------------------------------------------------------------------------------------------------###


###--------------------------------------------------------------------------------------------------###
categories = [labels_battery, labels_brakes, labels_doors, labels_engine_gearbox, labels_equipements,
             labels_interior_seats, labels_lights]
X_all_list = [X_battery, X_brakes, X_doors, X_engine_gearbox, X_equipements, X_interior_seats, X_lights]

s = 0
for data in categories:
    s += len(data)

X = np.zeros((s, 20, 128), dtype=np.float64)
Y = np.zeros((s, len(categories)), dtype=np.int32)

i = 0
for X_i in X_all_list:
    for slice in X_i:
        X[i, :, :] = slice
        i += 1

merged = list(itertools.chain(*categories))
for i, data_slice in enumerate(merged):
    if merged[i] == "battery":
        Y[i, 0] = 1
    elif merged[i] == "brakes":
        Y[i, 1] = 1
    elif merged[i] == "doors":
        Y[i, 2] = 1
    elif merged[i] == "engine_gearbox":
        Y[i, 3] = 1
    elif merged[i] == "equipements":
        Y[i, 4] = 1
    elif merged[i] == "interior_seats":
        Y[i, 5] = 1
    elif merged[i] == "lights":
        Y[i, 6] = 1
###--------------------------------------------------------------------------------------------------###

from sklearn.utils import shuffle
X, Y = shuffle(X, Y, random_state=0)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)


def sample(preds, temperature=1.0):
    return np.argmax(preds)
###-----------------------------------------------------------------------------------------------------###


###----------------------------------------------LSTM model---------------------------------------------###
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import keras.backend as K
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers import LSTM
from keras.optimizers import Adam

def f1(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2 * p * r / (p + r + K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)


def f1_loss(y_true, y_pred):
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2 * p * r / (p + r + K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return 1 - K.mean(f1)


def get_model_LSTM(X_train, Y_train, X_test, Y_test, batch_size, nb_epochs, lr):
    model = Sequential()
    model.add(LSTM(256, dropout=0.2, recurrent_dropout=0.2, input_shape=(20, 128),
                   return_sequences=False, go_backwards=True))

    model.add(Dense(128, activation='tanh'))

    model.add(Dense(len(categories), activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss=f1_loss, metrics=['accuracy', f1])

    # Fit the model
    model.fit(X_train, Y_train,
              batch_size=batch_size,
              shuffle=True,
              epochs=nb_epochs,
              validation_data=(X_test, Y_test),
              verbose=2)

    return model


model_LSTM = get_model_LSTM(X_train, Y_train, X_test, Y_test, 64, 100, 0.0001)
model_LSTM.save('LSTM.h5')
###-------------------------------------------------------------------------------------------###


###----------------------------------------------CONV1D model---------------------------------------------###
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import keras.backend as K
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.optimizers import Adam


def get_model_CONV1D(X_train, Y_train, X_test, Y_test, batch_size, nb_epochs, lr):
    # Keras convolutional model
    model = Sequential()
    model.add(Conv1D(32, kernel_size=5, activation='elu', padding='same', input_shape=(20, 128)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.25))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='elu'))
    model.add(Dropout(0.25))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=32, kernel_size=2, padding='same', activation='elu'))
    model.add(Dropout(0.25))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())

    model.add(Dense(128, activation='tanh'))
    model.add(Dense(7, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss=f1_loss, metrics=['accuracy', f1])

    # Fit the model
    model.fit(X_train, Y_train,
              batch_size=batch_size,
              shuffle=True,
              epochs=nb_epochs,
              validation_data=(X_test, Y_test),
              verbose=2)
    return model

model_CONV1D = get_model_CONV1D(X_train, Y_train, X_test, Y_test, 64, 100, 0.001)
model_CONV1D.save('CONV1D.h5')
###-------------------------------------------------------------------------------------------###


###--------------------------------------Predictions------------------------------------------###
project_name = "PHEV"
project_id = get_project_id_from_name(project_list, project_name)
query_list = get_query_id(project_id, access_token)
query_id = "1999796739"
stream_data("Test/PHEVs_brandWatch.csv", start_date, str(datetime.datetime.today().date()), project_id, query_id, access_token, new=True)
clean_brandwatch("Test/PHEVs_clean.csv", "Test/PHEVs_brandWatch.csv", "en", "", add=False, classe=False)

path = "Test"
flatten(path, path + "/" + "PHEVs_clean.csv", max_df=1.0, min_df=0.05)
X_test = top_n_keywords_concat_predict(word2vec_model, path + "/keywords.csv", path + "/coefs.csv", vector_size=128)

lables_list = ["battery", "brakes", "doors", "engine_gearbox", "equipements", "interior_seats", "lights"]
prediction = model.predict(X_test)
predictions_tab = []
for t in range(len(prediction)):
    predictions_tab.append(lables_list[sample(prediction[t])])
###-------------------------------------------------------------------------------------------###
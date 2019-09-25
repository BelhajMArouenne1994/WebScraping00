import itertools
import os
import pprint
from itertools import chain
from os import listdir
from os.path import isfile, join

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from Data_Processing.ReviewsSegmentation import *

###------------------------------------------------------------------------------------------------------------------###
stopWords = []
stopWords.append(stopwords.words('english'))
stopWords.append(stopwords.words('french'))
stopWords.append(stopwords.words('german'))
stopWords.append(["bmw", "tesla", "renault", "nissan", "peugeot", "audi", "citroen", "fiat", "kia", "volkswagen",
                  "toyota", "mazda", "jaguar", "mercedes", "benz", "model"])
stopWords = list(chain(*stopWords))

os.chdir(r"C:\Users\p100623\PycharmProjects\WebScraping\Models")
mypath = r"C:\Users\p100623\PycharmProjects\WebScraping\Models\TrainingDataSets"
onlyfiles = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f)) and f.find("clean") != -1]

corpus = []
for file in onlyfiles:
    for line in pd.read_csv(file, delimiter=',', error_bad_lines=False)["processedReviews"]:
        corpus.append(str(line).lower().split())
"""
word2vec_model = Word2Vec(corpus, size=256, window=5, min_count=20, workers=8, iter=10)
word2vec_model.save(r".\Word2vVec.model")
"""
###------------------------------------------------------------------------------------------------------------------###


###------------------------------------------------------------------------------------------------------------------###
mypath = r"C:\Users\p100623\PycharmProjects\WebScraping\Models"
word2vec_model = Word2Vec.load("Word2vVec.model")

print("Positive")
similar_words = {search_term: [item[0] for item in word2vec_model.wv.most_similar(positive=[search_term], topn=20)]
                 for search_term in [stemmer.stem("brake")]}
pprint.pprint(similar_words, compact=True)


###------------------------------------------------------------------------------------------------------------------###


###------------------------------------------------------------------------------------------------------------------###
def flatten(directory, processedReviews_file, max_df, min_df):
    try:
        os.mkdir(directory)
        print("Directory ", directory, " Created")
    except FileExistsError:
        print("Directory ", directory, " already exists")

    extract_tfidf_keywords(directory+r"\keywords.csv", processedReviews_file, 20, max_df, min_df)
    extract_tfidf_coefs(directory+r"\coefs.csv", processedReviews_file, 20, max_df, min_df)


###------------------------------------------------------------------------------------------------------------------###


###------------------------------------------------------------------------------------------------------------------###
path = r"TrainingDataSets\battery"
# flatten(path, r"TrainingDataSets\battery_clean.csv", max_df=1.0, min_df=0.001)
X_battery, labels_battery = top_n_keywords_concat("battery", word2vec_model, path+r"\keywords.csv",
                                                  path+r"\coefs.csv", vector_size=256)

path = r"TrainingDataSets\brakes"
# flatten(path, r"TrainingDataSets\brakes_clean.csv", max_df=1.0, min_df=0.001)
X_brakes, labels_brakes = top_n_keywords_concat("brakes", word2vec_model, path+r"\keywords.csv",
                                                path+r"\coefs.csv", vector_size=256)

path = r"TrainingDataSets\dealer"
# flatten(path, r"TrainingDataSets\dealer_clean.csv", max_df=1.0, min_df=0.001)
X_dealer, labels_dealer = top_n_keywords_concat("dealer", word2vec_model, path+r"\keywords.csv",
                                                path+r"\coefs.csv", vector_size=256)

path = r"TrainingDataSets\engine_gearbox"
# flatten(path, r"TrainingDataSets\engine_gearbox_clean.csv", max_df=1.0, min_df=0.001)
X_engine_gearbox, labels_engine_gearbox = top_n_keywords_concat("engine_gearbox", word2vec_model, path+r"\keywords.csv",
                                                                path+r"\coefs.csv", vector_size=256)

path = r"TrainingDataSets\equipements"
# flatten(path, r"TrainingDataSets\equipements_clean.csv", max_df=1.0, min_df=0.001)
X_equipements, labels_equipements = top_n_keywords_concat("equipements", word2vec_model, path+r"\keywords.csv",
                                                          path+r"\coefs.csv", vector_size=256)

path = r"TrainingDataSets\interior_seats"
# flatten(path, r"TrainingDataSets\interior_seats_clean.csv", max_df=1.0, min_df=0.001)
X_interior_seats, labels_interior_seats = top_n_keywords_concat("interior_seats", word2vec_model, path+r"\keywords.csv",
                                                                path+r"\coefs.csv", vector_size=256)

path = r"TrainingDataSets\price"
# flatten(path, r"TrainingDataSets\price_clean.csv", max_df=1.0, min_df=0.001)
X_price, labels_price = top_n_keywords_concat("price", word2vec_model, path+r"\keywords.csv",
                                              path+r"\coefs.csv", vector_size=256)
###------------------------------------------------------------------------------------------------------------------###


###--------------------------------------------------------------------------------------------------###
categories = [labels_battery, labels_brakes, labels_dealer, labels_engine_gearbox, labels_equipements,
              labels_interior_seats, labels_price]
X_all_list = [X_battery, X_brakes, X_dealer, X_engine_gearbox, X_equipements, X_interior_seats, X_price]

s = 0
for data in categories:
    s += len(data)

X = np.zeros((s, 20, 256), dtype=np.float64)
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
    elif merged[i] == "dealer":
        Y[i, 2] = 1
    elif merged[i] == "engine_gearbox":
        Y[i, 3] = 1
    elif merged[i] == "equipements":
        Y[i, 4] = 1
    elif merged[i] == "interior_seats":
        Y[i, 5] = 1
    elif merged[i] == "price":
        Y[i, 6] = 1


###--------------------------------------------------------------------------------------------------###


###--------------------------------------------------------------------------------------------------###
def sample(preds):
    lables_list = ["battery", "brakes", "dealer", "engine_gearbox", "equipements", "interior_seats", "price"]
    return lables_list[np.argmax(preds)]


X, Y = shuffle(X, Y, random_state=0)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
###-----------------------------------------------------------------------------------------------------###


###----------------------------------------------CONV1D model---------------------------------------------###
import tensorflow as tf
import keras.backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.models import load_model
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import mpu.ml


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


"""
def get_model_CONV1D(X_train, Y_train, X_test, Y_test, parameters):
    # Keras convolutional model
    model = Sequential()
    model.add(
        Conv1D(parameters['filters_1'], kernel_size=parameters['kernel_size_1'], activation=parameters['activation'],
               padding='same',
               input_shape=(20, 256)))
    model.add(MaxPooling1D(pool_size=2, strides=1))
    model.add(Dropout(parameters['dropout']))

    model.add(Conv1D(filters=parameters['filters_2'], kernel_size=parameters['kernel_size_2'],
                     activation=parameters['activation'],
                     padding='same'))
    model.add(MaxPooling1D(pool_size=2, strides=1))
    model.add(Dropout(parameters['dropout']))

    model.add(Conv1D(filters=parameters['filters_3'], kernel_size=parameters['kernel_size_3'],
                     activation=parameters['activation'],
                     padding='same'))
    model.add(MaxPooling1D(pool_size=2, strides=1))
    model.add(Dropout(parameters['dropout']))

    model.add(Flatten())
    model.add(Dense(256, activation='tanh'))
    model.add(Dense(7, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss=f1_loss, metrics=['accuracy', f1])

    early_stopping = EarlyStopping(monitor='val_acc',
                                   # go through epochs as long as accuracy on validation set increases
                                   patience=10,
                                   mode='max')

    # make sure that the model corresponding to the best epoch is saved
    checkpointer = ModelCheckpoint(filepath='CONV1D.h5',
                                   monitor='val_acc',
                                   save_best_only=True,
                                   verbose=0)

    # Fit the model
    out = model.fit(X_train, Y_train,
                    batch_size=parameters['batch_size'],
                    shuffle=True,
                    epochs=parameters['nb_epochs'],
                    validation_data=(X_test, Y_test),
                    verbose=2,
                    callbacks=[early_stopping, checkpointer])
    return out, model


p = {'filters_1': [30, 60, 120, 240],
     'filters_2': [30, 60, 120, 240],
     'filters_3': [30, 60, 120, 240],
     'kernel_size_1': [1, 2, 3, 4],
     'kernel_size_2': [1, 2, 3, 4],
     'kernel_size_3': [1, 2, 3, 4],
     'batch_size': [64, 128, 256],
     'nb_epochs': [100, 200, 300],
     'dropout': [0.05, 0.1, 0.2],
     'optimizer': ['Adam'],
     'activation': ['relu', 'elu']}

h = Scan(X_train,
         Y_train,
         model=get_model_CONV1D,
         params=p,
         print_params=True,
         experiment_name='256-5-20')
"""


def get_model_CONV1D(X_train, Y_train, X_test, Y_test, batch_size, nb_epochs):
    # Keras convolutional model
    model = Sequential()
    model.add(Conv1D(150, kernel_size=3, activation='relu', padding='same', input_shape=(20, 256)))
    model.add(MaxPooling1D(pool_size=2, strides=1))
    model.add(Dropout(0.1))

    model.add(Conv1D(filters=150, kernel_size=4, activation='relu', padding='same'))
    model.add(MaxPooling1D(pool_size=2, strides=1))
    model.add(Dropout(0.1))

    model.add(Flatten())
    model.add(Dense(256, activation='tanh'))
    model.add(Dense(7, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss=f1_loss, metrics=['accuracy', f1])

    early_stopping = EarlyStopping(monitor='val_acc',
                                   # go through epochs as long as accuracy on validation set increases
                                   patience=10,
                                   mode='max')

    # make sure that the model corresponding to the best epoch is saved
    checkpointer = ModelCheckpoint(filepath='CONV1D.h5',
                                   monitor='val_acc',
                                   save_best_only=True,
                                   verbose=0)

    # Fit the model
    model.fit(X_train, Y_train,
              batch_size=batch_size,
              shuffle=True,
              epochs=nb_epochs,
              validation_data=(X_test, Y_test),
              verbose=2,
              callbacks=[early_stopping, checkpointer])
    return model


model_CONV1D = get_model_CONV1D(X_train, Y_train, X_test, Y_test, 256, 200)
model_CONV1D.save('CONV1D_2.h5')
model_CONV1D = load_model('CONV1D.h5', custom_objects={'f1': f1, 'f1_loss': f1_loss})
model_CONV1D.summary()
###-------------------------------------------------------------------------------------------###


###-------------------------------------------------------------------------------------------###
get_doc_embedding = K.function([model_CONV1D.layers[0].input, K.learning_phase()], [model_CONV1D.layers[8].output])
n_plot = 1400
print('plotting embeddings of first', n_plot, 'documents')

doc_emb = get_doc_embedding([np.array(X_test[:n_plot]), 0])[0]
nsamples, nx, ny = doc_emb.shape
d2_train_dataset = doc_emb.reshape((nsamples, nx*ny))

my_pca = PCA(n_components=10)
my_tsne = TSNE(n_components=3, perplexity=10)
doc_emb_pca = my_pca.fit_transform(d2_train_dataset)
doc_emb_tsne = my_tsne.fit_transform(doc_emb_pca)

labels_plt = mpu.ml.one_hot2indices(Y_test[:n_plot])
my_colors = ["blue", "red", "green", "gray", "black", "pink", "yellow"]

fig, ax = plt.subplots()

for label in list(set(labels_plt)):
    idxs = [idx for idx, elt in enumerate(labels_plt) if elt == label]
    ax.scatter(doc_emb_tsne[idxs, 0],
               doc_emb_tsne[idxs, 1],
               c=my_colors[label],
               label=str(label),
               alpha=0.7,
               s=10)

ax.legend(scatterpoints=1)
fig.suptitle('t-SNE visualization of CNN-based doc embeddings \n (first 1000 docs from test set)', fontsize=10)
fig.set_size_inches(12, 8)
fig.savefig('doc_embeddings.png', bbox_inches='tight')
fig.show()
###-------------------------------------------------------------------------------------------###

###--------------------------------------Predictions------------------------------------------###
prediction = model_CONV1D.predict(X_test)
predictions_tab = []
for t in range(len(prediction)):
    predictions_tab.append(sample(prediction[t]))

y_test = []
for i in range(len(Y_test)):
    if Y_test[i][0] == 1:
        y_test.append("battery")
    elif Y_test[i][1] == 1:
        y_test.append("brakes")
    elif Y_test[i][2] == 1:
        y_test.append("dealer")
    elif Y_test[i][3] == 1:
        y_test.append("engine_gearbox")
    elif Y_test[i][4] == 1:
        y_test.append("equipements")
    elif Y_test[i][5] == 1:
        y_test.append("interior_seats")
    elif Y_test[i][6] == 1:
        y_test.append("price")

from mlxtend.evaluate import confusion_matrix

cm = confusion_matrix(y_target=y_test,
                      y_predicted=predictions_tab,
                      binary=False)

import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix

class_names = ["battery", "brakes", "dealer", "engine_gearbox", "equipements", "interior_seats", "price"]
fig, ax = plot_confusion_matrix(conf_mat=cm,
                                show_absolute=True,
                                show_normed=True,
                                colorbar=True,
                                class_names=class_names)
plt.show()
plt.savefig('confusion_matrix.png', bbox_inches='tight')

import gensim
import itertools
import multiprocessing
import pandas as pd
from itertools import *


class HyperparameterError(Exception):
    """Erreur liée aux hyperparamètres passés en entrée"""
    pass


class UneditableError(Exception):
    """Attribut non modifiable"""
    pass


class GridSearchWord2Vec:
    """Résultat du grid search du word2vec
    """

    def __init__(self, model, type_msr, msr, hyperparameters, vocab):
        """Constructeur de la classe"""
        self._model = model
        self._type_msr = type_msr
        self._msr = msr
        self._hyperparameters = hyperparameters
        self._vocab = vocab

    def __delattr__(self, nom_attr):
        """On ne peut supprimer d'attribut, on lève l'exception AttributeError"""
        raise AttributeError("Vous ne pouvez supprimer aucun attribut de cette classe")

    def __ne__(self, autre):
        if self.type_msr != autre.type_msr:
            raise TypeError("Problème de mesure")
        else:
            return self.msr != autre.msr

    def __eq__(self, autre):
        if self.type_msr != autre.type_msr:
            raise TypeError("Problème de mesure")
        else:
            return self.msr == autre.msr

    def __le__(self, autre):
        if self.type_msr != autre.type_msr:
            raise TypeError("Problème de mesure")
        else:
            return self.msr <= autre.msr

    def __lt__(self, autre):
        if self.type_msr != autre.type_msr:
            raise TypeError("Problème de mesure")
        else:
            return self.msr < autre.msr

    def __ge__(self, autre):
        if self.type_msr != autre.type_msr:
            raise TypeError("Problème de mesure")
        else:
            return self.msr >= autre.msr

    def __gt__(self, autre):
        if self.type_msr != autre.type_msr:
            raise TypeError("Problème de mesure")
        else:
            return self.msr > autre.msr

    def __str__(self):
        return ("Word2Vec(min_count={mc}, size={s}, window={w}, iter={i}, seed={sd})".format(
            mc=self.hyperparameters["min_count"],
            s=self.hyperparameters["size"],
            w=self.hyperparameters["window"],
            i=self.hyperparameters["iter"],
            sd=self.hyperparameters["seed"]))

    def _set_model(self, model):
        raise UneditableError("On ne modifie pas ce paramètre")

    def _get_model(self):
        return self._model

    def _set_type_msr(self, type_msr):
        raise UneditableError("On ne modifie pas ce paramètre")

    def _get_type_msr(self):
        return self._type_msr

    def _set_msr(self, msr):
        raise UneditableError("On ne modifie pas ce paramètre")

    def _get_msr(self):
        return self._msr

    def _set_hyperparameters(self, hyperparameters):
        raise UneditableError("On ne modifie pas ce paramètre")

    def _get_hyperparameters(self):
        return self._hyperparameters

    def _set_vocab(self, vocab):
        raise UneditableError("On ne modifie pas ce paramètre")

    def _get_vocab(self):
        return self._vocab

    hyperparameters = property(_get_hyperparameters, _set_hyperparameters)
    msr = property(_get_msr, _set_msr)
    type_msr = property(_get_type_msr, _set_type_msr)
    model = property(_get_model, _set_model)
    vocab = property(_get_vocab, _set_vocab)


def score_w2v(model):
    similarite = []
    list_words = [["ev", []]
                  ["vehicle", ["car", "automobile", "truck"]],
                  ["drive", []],
                  ["brake", []],
                  ["battery", ["rechargeable", "charger", "lithium", "ion", "Lithium-ion"]],
                  ["plug-in", ["connector"]],
                  ["hybrid", ["dual"]],
                  ["electric", []],
                  ["HEV", []],
                  ["PHEV", []],
                  ["Fuel", []],
                  ["charging", []],
                  ["connector", []],
                  ["home", []],
                  ["phase", []]]
    for i in range(len(list_words)):
        word_1 = list_words[i][0]
        for j in range(len(list_words[i])):
            if j>0:
                word_2 = list_words[i][j]
            similarite.append(model.similarity(word_1, word_2))
    return(sum(similarite))


def grid_word2vec(dict_hyper_param, tokens, func_q=score_w2v):
    '''Grid search adapté au word2vec

    @param: dict_hyper_param dictionnaire ayant pour clefs les paramètres de word2vec inclues dans ["min_count", "size", "window", "iter", "seed"]
    @param: tokens listes de listes de mots sur lesquelles entrainer le modèle
    @param: func_q fonction de mesure de la qualité

    @return: meilleur modèle
    '''
    keys_hp = dict_hyper_param.keys()
    if not all([key in ["min_count", "size", "window", "iter", "seed"] for key in keys_hp]):
        raise HyperparameterError("Les clefs passées dans le dictionnaire en entrée ne sont pas valides")
    else:
        list_param = []
        if not "min_count" in keys_hp:
            dict_hyper_param["min_count"] = [20]
        if not "window" in keys_hp:
            dict_hyper_param["window"] = [5]
        if not "iter" in keys_hp:
            dict_hyper_param["iter"] = [15]
        if not "size" in keys_hp:
            dict_hyper_param["size"] = [128]
        if not "seed" in keys_hp:
            dict_hyper_param["seed"] = [1234]
        dict_num_key = dict()  # dictionnaire contenant les indices des clefs pour affecter le bon paramètre
        for i, key in enumerate(["min_count", "size", "window", "iter", "seed"]):
            dict_num_key[key] = i
            list_param.append(dict_hyper_param[key])
        combi = list(itertools.product(*list_param))  # construction des combinaisons de paramètres
        best_msr = 0
        best_model = -1
        best_combi = -1
        for c in combi:
            model = gensim.models.Word2Vec(sentences=tokens,
                                 min_count=c[dict_num_key["min_count"]],
                                 size=c[dict_num_key["size"]],
                                 window=c[dict_num_key["window"]],
                                 iter=c[dict_num_key["iter"]],
                                 seed=c[dict_num_key["seed"]],
                                 workers=multiprocessing.cpu_count())
            msr = func_q(model)
            if msr > best_msr:
                best_msr = msr
                best_model = model
                best_combi = dict()
                for key in ["min_count", "size", "window", "iter", "seed"]:
                    best_combi[key] = c[dict_num_key[key]]
        type_msr = ""
        return GridSearchWord2Vec(best_model, type_msr, best_msr, best_combi, tokens)


class corpus(object):
    def __iter__(self):
        for line in pd.read_csv('intermediate/processedReviews2.csv', delimiter=',', error_bad_lines=False, encoding="utf-8")["processedReviews"]:
            yield line.lower().split()


model = grid_word2vec([[10, 20, 30], [64, 128, 256], [15, 20, 30]], corpus(), func_q=score_w2v())
import json
import pymongo
import Data_Processing.Review as RW
import Data_Processing.MongoData as MongoData
import datetime
import pandas as pd
import os
import shutil


class ReviewManagement(object):
    """manages reviewlass objects"""
    revList = []

    def _init_(self):
        revList = self.revList

    def populateReviews(self):
        md = MongoData.MongoData()
        cursor = md.getCursorAllCars()
        for document in cursor:
            self.jsonToObject(document)

    # method for converting json to objects
    def jsonToObject(self, prod):
        try:
            ID = prod["ID"]
            Car = prod["Car"]
            Country = prod["Country"].replace("'", "").replace("{", "").replace("}", "").replace(" ", "_")
            Text = prod["Text"]
            Date = prod["Date"]
            Page_Type = prod["Page_Type"]
            ReviewObj = RW.Review(ID, Car, Country, Text, Date, Page_Type)
            self.revList.append(ReviewObj)
        except:
            pass

    # method for returning product names
    def getReviews(self):
        reviews = []
        for index in range(len(self.revList)):
            reviews.append(self.revList[index])
        return reviews

    def getCountries(self):
        countries = []
        for index in range(len(self.revList)):
            countries.append(self.revList[index].Country)
        return set(countries)

    def getCars(self):
        cars = []
        for index in range(len(self.revList)):
            cars.append(self.revList[index].Car.replace("'", ""))
        return set(cars)

    def getReviewsByCar(self, carName, country, page_type_list, mot_cle, start_date, end_date):
        directory = r"Outputs\."
        try:
            os.mkdir(directory)
        except FileExistsError:
            shutil.rmtree(directory)
            os.mkdir(directory)

        directory = r"Outputs\DataSets\."
        try:
            os.mkdir(directory)
        except FileExistsError:
            pass

        carName = carName.replace("'", "")
        with open(r"Outputs\DataSets\Dataset.csv", 'a+', encoding='utf-8') as f:
            f.write("carµCountryµTextµDate")
            f.write("\n")
        f.close()

        for index in range(len(self.revList)):
            try:
                def clean_int(x):
                    if x[0] == "0":
                        return x[1]
                    else:
                        return x

                if self.revList[index].Car == carName and \
                        str(self.revList[index].Text).find(mot_cle) != -1 and (self.revList[index].Country == country or
                                                                               country == "ALL") and \
                        str(self.revList[index].Page_Type) in page_type_list \
                        and datetime.date(int(start_date.split("-")[0]), int(clean_int(start_date.split("-")[1])),
                                          int(clean_int(start_date.split("-")[2]))) <= \
                        datetime.date(int(self.revList[index].Date.split("-")[0]),
                                      int(clean_int(self.revList[index].Date.split("-")[1])),
                                      int(clean_int(self.revList[index].Date.split("-")[2]))) <= \
                        datetime.date(int(end_date.split("-")[0]), int(clean_int(end_date.split("-")[1])),
                                      int(clean_int(end_date.split("-")[2]))):
                    with open(r"Outputs\DataSets\Dataset.csv", 'a+', encoding='utf-8') as f:
                        f.write(self.revList[index].Car)
                        f.write("µ")
                        f.write(self.revList[index].Country)
                        f.write("µ")
                        f.write(self.revList[index].Text.replace('"', " ").replace("\n", " ").replace("\r", " "))
                        f.write("µ")
                        f.write(self.revList[index].Date)
                        f.write("\n")
                    f.close()
            except:
                pass

        message = ""
        try:
            data = pd.read_csv(r"Outputs\DataSets\Dataset.csv", error_bad_lines=False, encoding='utf-8', delimiter="µ",
                               engine='python')
            if data.shape[0] > 0:
                message = "Taille de la Base de données = "+str(data.shape[0])+"."
            elif data.shape[0] == 0:
                message = "Aucun élement correspondant à vos critères dans la Base de données."
        except:
            message = "Erreur lors du processus de contruction de la base."

        return message

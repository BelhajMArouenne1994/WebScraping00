from pymongo import MongoClient


class MongoData(object):
    """Handles all Mongo transactions"""
    client = MongoClient(host="localhost", port=27017)
    db = client.Renault

    def _init_(self):
        self.client = MongoClient(host="localhost", port=27017)
        self.db = self.client.Renault

    # get all products
    def getCursorAllCars(self):
        cursor = self.db.cars.find()
        self.client.close()
        return cursor

    # get products based on some criteria
    def getCursorCarsCriteria(self, criteria):
        cursor = self.db.cars.find(criteria)
        self.client.close()
        return cursor

    # get last _id
    def getCountReviews(self):
        result = self.db.cars.count()
        return result

    # insert a new Review
    def insertReview(self, document):
        result = self.db.cars.insert_one(document)
        self.client.close()
        return result

    # return reviews based on product
    def getReviewbyCar(self, productName):
        criteria = {"Car": productName}
        cursor = self.db.cars.find(criteria)
        return cursor

from flask import Flask
from flask_pymongo import pymongo
CONNECTION_STRING = "mongodb+srv://sajinaaron07:315sajMay.2@cluster0.ryjqylk.mongodb.net/?retryWrites=true&w=majority"
client = pymongo.MongoClient(CONNECTION_STRING)
db = client.get_database('Stroke')
user_collection = pymongo.collection.Collection(db, 'user_collection')
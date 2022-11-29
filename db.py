from flask import Flask
from flask_pymongo import pymongo
CONNECTION_STRING = "mongodb+srv://potter:potter123@api.0az9lg4.mongodb.net/?retryWrites=true&w=majority"
client = pymongo.MongoClient(CONNECTION_STRING)
db = client.get_database('flask_mongodb_atlas')
user_collection = pymongo.collection.Collection(db, 'user_collection')
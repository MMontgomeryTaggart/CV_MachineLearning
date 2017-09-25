from pymongo import MongoClient
import numpy as np

#get a list of all the notes
client = MongoClient()
collection = client["NLP"]["NoteTracking"]

result = collection.aggregate([{"$match" : {"human_annotation" : True}}, {"$group" : { "_id" : 0, "notes" : {"$addToSet" : "$name"}}}])

resultObj = next(result)
nameList = resultObj["notes"]

nameList = np.array(nameList)

firstThird = np.random.choice(nameList, size=320, replace=False)

remainder = np.setdiff1d(nameList, firstThird)

secondThird = np.random.choice(remainder, size=320, replace=False)

thirdThird = np.setdiff1d(remainder, secondThird)

assert len(np.setdiff1d(firstThird, secondThird)) == 320
assert len(np.setdiff1d(firstThird, thirdThird)) == 320
assert len(np.setdiff1d(secondThird, thirdThird)) == 320

assert len(firstThird) == 320
assert len(secondThird) == 320
assert len(thirdThird) == 320

data = [(firstThird, "train"), (secondThird, "test"), (thirdThird, "target")]

for datum in data:
    collection.update_many({"name" : {"$in" : datum[0]}}, {"$set" : {"learning_group" : datum[1]}})


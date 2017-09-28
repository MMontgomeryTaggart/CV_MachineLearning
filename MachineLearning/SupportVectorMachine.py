from sklearn import svm
from sklearn import metrics
import numpy as np
from pymongo import MongoClient
import re

client = MongoClient()

# ************************  Obtain training data and fit the model ********************
noteTracking = client["NLP"]["NoteTracking"]
trainingDocsResult = noteTracking.aggregate([{
    "$match" : {"learning_group" : "train"}
    },
    {"$group" : {"_id" : 1, "names" : {"$addToSet" : "$name"}}}

])

trainingDocsFirstResult = next(trainingDocsResult)
trainingDocNames = trainingDocsFirstResult['names']


annotationsCollection = client["NLP"]["Annotations"]
docClassificationResults = annotationsCollection.aggregate([
    {"$match" : {"document_name" : {"$in" : trainingDocNames}}},
    {"$unwind" : "$annotations"},
    {"$match" : {"annotations.text" : "DOC CLASS"}},
    {"$group" : {"_id" : "$annotations.attributes.present_or_absent", "names" : {"$addToSet" : "$document_name"}}}
])

docNames = []
truthVec = []

for result in docClassificationResults:
    for doc in result["names"]:
        docNames.append(doc)
        if result["_id"] == "present":
            truthVec.append(1)
        else:
            truthVec.append(0)

docBodies = []
numFeatures = 6
features = np.zeros((len(docNames), numFeatures))

corpusPath = "/users/shah/Box Sync/MIMC_v2/Corpus/corpus/corpus/"
for name in docNames:
    fullPath = corpusPath + name
    with open(fullPath + ".txt", 'rU') as inFile:
        docBodies.append(inFile.read())


# Features:
# 1. count of regex "bleed".
# 2. count of regex: "hemorrhage" variants.
# 3. Count of hematoma.
# 4. Count of regex "coffee\s+ground".
# 5. Count of hematuria
# 6. Count of regex "melena" variants.

for index, body in enumerate(docBodies):
    bleedCount = len(re.findall("bleed", body, flags=re.IGNORECASE))
    hemmorrhageCount = len(re.findall("hem+or+h*age", body, flags=re.IGNORECASE))
    hematomaCount = len(re.findall("hematoma", body, flags=re.IGNORECASE))
    coffeeGroundsCount = len(re.findall("coffee\s+ground", body, flags=re.IGNORECASE))
    hematuriaCount = len(re.findall("hematuria", body, flags=re.IGNORECASE))
    melanaCount = len(re.findall("mel[ae]n(a|ic)", body, flags=re.IGNORECASE))

    featureVec = np.array(
        [bleedCount, hemmorrhageCount, hematomaCount, coffeeGroundsCount, hematuriaCount, melanaCount])

    features[index] = featureVec

svmModel = svm.SVC()
svmModel.fit(features, truthVec)

# ******************* Obtain Test Data and Evaluate Model ***********************

testDocsResult = noteTracking.aggregate([{
    "$match" : {"learning_group" : "test"}
    },
    {"$group" : {"_id" : 1, "names" : {"$addToSet" : "$name"}}}

])

testDocsFirstResult = next(testDocsResult)
testDocNames = testDocsFirstResult['names']

testDocClassificationResults = annotationsCollection.aggregate([
    {"$match" : {"document_name" : {"$in" : testDocNames}}},
    {"$unwind" : "$annotations"},
    {"$match" : {"annotations.text" : "DOC CLASS"}},
    {"$group" : {"_id" : "$annotations.attributes.present_or_absent", "names" : {"$addToSet" : "$document_name"}}}
])

client.close()

testDocNames = []
testTruthVec = []

for result in testDocClassificationResults:
    for doc in result["names"]:
        testDocNames.append(doc)
        if result["_id"] == "present":
            testTruthVec.append(1)
        else:
            testTruthVec.append(0)

testDocBodies = []
testFeatures = np.zeros((len(docNames), numFeatures))

corpusPath = "/users/shah/Box Sync/MIMC_v2/Corpus/corpus/corpus/"
for name in testDocNames:
    fullPath = corpusPath + name
    with open(fullPath + ".txt", 'rU') as inFile:
        testDocBodies.append(inFile.read())


for index, body in enumerate(testDocBodies):
    bleedCount = len(re.findall("bleed", body, flags=re.IGNORECASE))
    hemmorrhageCount = len(re.findall("hem+or+h*age", body, flags=re.IGNORECASE))
    hematomaCount = len(re.findall("hematoma", body, flags=re.IGNORECASE))
    coffeeGroundsCount = len(re.findall("coffee\s+ground", body, flags=re.IGNORECASE))
    hematuriaCount = len(re.findall("hematuria", body, flags=re.IGNORECASE))
    melanaCount = len(re.findall("mel[ae]n(a|ic)", body, flags=re.IGNORECASE))

    featureVec = np.array(
        [bleedCount, hemmorrhageCount, hematomaCount, coffeeGroundsCount, hematuriaCount, melanaCount])

    testFeatures[index] = featureVec

predictions = svmModel.predict(testFeatures)

print metrics.precision_recall_fscore_support(testTruthVec, predictions, average="binary")
print "hi"
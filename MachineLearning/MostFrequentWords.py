from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import re

client = MongoClient()

annotationsCollection = client["NLP"]["Annotations"]

results = annotationsCollection.aggregate([
    {'$unwind': "$annotations"},
    {"$match" : {"annotations.text" : "DOC CLASS"}},
    {"$group":
         {"_id" :
              {"bleeding" : "$annotations.attributes.present_or_absent"},
          "documents" : {"$addToSet": "$document_name"}}}])
client.close()

resultDict = {}

for result in results:
    resultDict[result["_id"]["bleeding"]] = result["documents"]

docNames = resultDict["present"] + resultDict["absent"]
corpusPath = "/Volumes/Fresh Apples/Box Sync/MIMC_v2/Corpus/corpus/corpus/"

presentBodies = []
absentBodies = []

for name in docNames:
    with open(corpusPath + name + ".txt", 'rU') as inFile:
        text = inFile.read()
        if name in resultDict["present"]:
            presentBodies.append(text)
        else:
            absentBodies.append(text)


presentVectorizer = CountVectorizer(stop_words='english')
absentVectorizer = CountVectorizer(stop_words='english')

digitPattern = re.compile(r"\b\d+")

presentMatrixWithDigits = presentVectorizer.fit_transform(presentBodies)
presentVocabWithDigits = presentVectorizer.get_feature_names()

presentIndicesToKeep = [index for index, word in enumerate(presentVocabWithDigits) if not digitPattern.match(word)]
presentVocab = map(lambda i: presentVocabWithDigits[i], presentIndicesToKeep)
presentMatrix = presentMatrixWithDigits[:, presentIndicesToKeep]

absentMatrixWithDigits = absentVectorizer.fit_transform(absentBodies)
absentVocabWithDigits = absentVectorizer.get_feature_names()

absentIndicesToKeep = [index for index, word in enumerate(absentVocabWithDigits) if not digitPattern.match(word)]
absentVocab = map(lambda i: absentVocabWithDigits[i], absentIndicesToKeep)
absentMatrix = absentMatrixWithDigits[:, absentIndicesToKeep]

presentMeans = np.array(np.mean(presentMatrix, axis=0))[0]
absentMeans = np.array(np.mean(absentMatrix, axis=0))[0]

top10PresentTuples = sorted(enumerate(presentMeans), key=lambda x: x[1], reverse=True)[:9]
top10AbsentTuples = sorted(enumerate(absentMeans), key=lambda x: x[1], reverse=True)[:9]

top10PresentIndices = list(zip(*top10PresentTuples))[0]
top10AbsentIndices = list(zip(*top10AbsentTuples))[0]



print "Present Words:"
print map(lambda index: presentVocab[index], top10PresentIndices)

print "Absent Words:"
print map(lambda index: absentVocab[index], top10AbsentIndices)

from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import numpy as np
import re

client = MongoClient()

noteTracking = client["NLP"]["NoteTracking"]
trainingDocsResult = noteTracking.aggregate([
    {"$match" : {"learning_group" : "train"}},
    {"$group" : {"_id" : 0, "names" : {"$addToSet" : "$name"}}}
])
firstTrainingDocResult = next(trainingDocsResult)

trainingDocs = firstTrainingDocResult["names"]
print "%i Training Docs" % len(trainingDocs)

annotationsCollection = client["NLP"]["Annotations"]
results = annotationsCollection.aggregate([
    {"$match" : {"document_name" : {"$in" : trainingDocs}}},
    {"$unwind": "$annotations"},
    {"$match" : {"annotations.text" : "DOC CLASS"}},
    {"$group":
         {"_id" :
              {"bleeding" : "$annotations.attributes.present_or_absent"},
          "documents" : {"$addToSet": "$document_name"}
          }
     }
])

client.close()

resultDict = {}

for result in results:
    resultDict[result["_id"]["bleeding"]] = result["documents"]

docNames = resultDict["present"] + resultDict["absent"]
corpusPath = "/users/shah/Box Sync/MIMC_v2/Corpus/corpus/corpus/"

bodies = []
presentVector = []

for name in docNames:
    with open(corpusPath + name + ".txt", 'rU') as inFile:
        bodies.append(inFile.read())
    if name in resultDict["present"]:
        presentVector.append(1)
    else:
        presentVector.append(0)

vectorizer = CountVectorizer(ngram_range=(1, 3), stop_words='english')
vectorsWithDigitFeatures = vectorizer.fit_transform(bodies)
vocabWithDigitFeatures = vectorizer.get_feature_names()

#Most of the top words are digit-strings, e.g. '2591', '4592', '138', etc. Let's remove those from the
# features before selection and fitting.
pattern = re.compile(r"\b\d+\b")
nonDigitColumns = [index for index, word in enumerate(vocabWithDigitFeatures) if not pattern.match(word)]

vectors = vectorsWithDigitFeatures[:, nonDigitColumns]
vocab = [vocabWithDigitFeatures[index] for index in nonDigitColumns]

assert vectors.shape[1] == len(vocab)

topKFeatures = SelectKBest(chi2, k=5)
x_new = topKFeatures.fit_transform(vectors, presentVector)

top50Scores = sorted(enumerate(topKFeatures.scores_), key=lambda x: x[1], reverse=True)[:49]
top50Indices = map(list, zip(*top50Scores))[0]

outFile = open("./output/Top50CountVectorizerTrainingOnly.txt", 'w')
outFile.write("Word\tP-Value\tScore\n")

for index, word in enumerate(map(lambda x: vocab[x], top50Indices)):
    outFile.write(word + '\t' + str(topKFeatures.pvalues_[top50Indices[index]]) + '\t' + str(topKFeatures.scores_[top50Indices[index]]) + '\n')

outFile.close()


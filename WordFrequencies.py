from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import re

client = MongoClient()

# Query a list of present and absent docs:
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

#Read the note bodies in as lists of strings:
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

presentFrequenciesBleed = np.zeros(len(presentBodies))
presentFrequenciesHemorrhage = np.zeros(len(presentBodies))
absentFrequenciesBleed = np.zeros(len(absentBodies))
absentFrequenciesHemorrhage = np.zeros(len(absentBodies))

def calculateFrequencies(notes, bleedFrequencies, hemorrhageFrequencies):
    for index, text in enumerate(notes):
        bleedFrequencies[index] = len(re.findall(r"bleed", text))
        hemorrhageFrequencies[index] = len(re.findall(r"hemorrhage", text))

    return bleedFrequencies, hemorrhageFrequencies

presentFrequenciesBleed, presentFrequenciesHemorrhage = calculateFrequencies(presentBodies, presentFrequenciesBleed, presentFrequenciesHemorrhage)
absentFrequenciesBleed, absentFrequenciesHemorrhage = calculateFrequencies(absentBodies, absentFrequenciesBleed, absentFrequenciesHemorrhage)

print "Mean bleed frequency (Present): %f" % np.mean(presentFrequenciesBleed)
print "Total bleed mentions (present) : %i, Total Bleed Docs: %i" % (np.sum(presentFrequenciesBleed), len(presentBodies))
print "Mean bleed frequency (Absent): %f" % np.mean(absentFrequenciesBleed)
print "Total bleed mentions (Absent) : %i, Total Bleed_absent Docs: %i" % (np.sum(absentFrequenciesBleed), len(absentBodies))
print "Mean hemorrhage frequency(Present): %f" % np.mean(presentFrequenciesHemorrhage)
print "Total hemorrhage mentions (present) : %i, Total hemorrhage Docs: %i" % (np.sum(presentFrequenciesHemorrhage), len(presentBodies))
print "Mean hemorrhage frequency(Absent): %f" % np.mean(absentFrequenciesHemorrhage)
print "Total hemorrhage mentions (present) : %i, Total hemorrhage Docs: %i" % (np.sum(absentFrequenciesHemorrhage), len(absentBodies))


#Results:
# 955 Notes were returned from the query. And there are 960 documents in the database. So 5 are not as expected, I'll look into it.
#
# Mean bleed frequency (Present): 1.857798
# Mean bleed frequency (Absent): 0.294437
# Mean hemorrhage frequency(Present): 0.344037
# Mean hemorrhage frequency(Absent): 0.047490
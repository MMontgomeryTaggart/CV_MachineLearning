from pymongo import MongoClient
import numpy as np
from eHostess.eHostInterface.KnowtatorReader import KnowtatorReader
from eHostess.MongoDBInterface.MongoTools import InsertMultipleDocuments
import pickle

def determineDocClass(doc):
    for annotation in doc.annotations:
        if annotation.annotationClass == "doc_classification":
            if annotation.attributes["present_or_absent"] == "present":
                return 1
            else:
                return 0

    return "?"

#Annotate the test knowtator files and import them into mongo
batchRange = range(33, 54 + 1)
docs = []
for batchNo in batchRange:
    notesPath = "/Users/shah/Developer/ShahNLP/TestNotes/batches/batch_%i/saved/*" % batchNo
    docs.extend(KnowtatorReader.parseMultipleKnowtatorFiles(notesPath))
# docNames = []
# for doc in docs:
#     docNames.append(doc.documentName)
#
# pickle.dump(docNames, open("./DocumentNames.pkl", 'wb'))
# print InsertMultipleDocuments(docs, "MIMC_v2")


presentDocs = []
absentDocs = []
noAnnotationDocs = []

for doc in docs:
    if determineDocClass(doc) == 1:
        presentDocs.append(doc.documentName)
    elif determineDocClass(doc) == 0:
        absentDocs.append(doc.documentName)
    else:
        noAnnotationDocs.append(doc.documentName)


print "Length of present docs: %i" % len(presentDocs)
print "Length of absent docs: %i" % len(absentDocs)

with open("../DocumentClasses.txt", "w") as outFile:
    for doc in noAnnotationDocs:
        outFile.write("%s\t%s\n"%(doc, '?'))
    for doc in presentDocs:
        outFile.write("%s\t%i\n"%(doc, 1))
    for doc in absentDocs:
        outFile.write("%s\t%i\n"%(doc, 0))

# def cleanNoteName(name):
#     parts = name.split("/")
#     return parts[-1].split(".")[0]
#
# def dbNoteName(name):
#     return name.split('/')[-1]
#
# notePath = "/users/shah/Box Sync/MIMC_v2/Annotation/Shane/batch_9/corpus/*"
# noteNames = glob.glob(notePath)
#
# noteTracking = client["NLP"]["NoteTracking"]
#
# for name in noteNames:
#     cleanName = cleanNoteName(name)
#     dbName = dbNoteName(name)
#
#     noteTracking.update_one({"name" : dbName}, {"$set" : {"name" : cleanName}})



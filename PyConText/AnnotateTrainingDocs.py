import eHostess.PyConTextInterface.PyConText as PyConText
import eHostess.PyConTextInterface.SentenceSplitters.TargetSpanSplitter as SpanSplitter
import pyConTextNLP.itemData as itemData
import site
from pymongo import MongoClient
import glob
import re
import json

def convertAnnotationToCSVString(annotation, documentLength, documentName, documentClassification):
    if annotation == None:
        #header
        return "DocumentName\tTrueDocumentClass\tAnnotationText\tPredictedAnnotationClass\tTarget\tModifier(s)\tDocLength\tAnnotationStart\tAnnotationEnd\n"
    annotationText = annotation.text
    annotationText = re.sub("\s+", " ", annotationText)
    predictedClass = 1
    if annotation.annotationClass in ["bleeding_absent", "bleeding_historical", "bleeding_hypothetical"]:
        predictedClass = 0
    target = annotation.dynamicProperties["target"]
    modifiers = str(annotation.dynamicProperties["modifiers"])

    return "%s\t%i\t%s\t%i\t%s\t%s\t%i\t%i\t%i\n" % (documentName,
                                                     documentClassification,
                                                     annotationText,
                                                     predictedClass,
                                                     target,
                                                     modifiers,
                                                     documentLength,
                                                     annotation.start,
                                                     annotation.end)

notesPath = "/users/shah/Box Sync/MIMC_v2/Corpus_TrainTest/"

def cleanName(name):
    file = name.split('/')[-1]
    return file.split('.')[0]

# get the list of positive documents from Mongo
docNames = glob.glob(notesPath + "*")
cleanNames = map(cleanName, docNames)

client = MongoClient()
annotations = client["NLP"]["Annotations"]

result = annotations.aggregate([
    {"$match" : {"document_name" : {"$in" : cleanNames}}},
    {"$unwind" : "$annotations"},
    {"$match" : {"annotations.text" : "DOC CLASS", "annotations.attributes.present_or_absent" : "present"}},
    {"$group" : {"_id" : 0, "names" : {"$addToSet" : "$document_name"}}}
])

client.close()

firstResult = next(result)
positiveDocs = firstResult['names']
# notesPath = "/users/shah/Developer/testNotes/corpus/"

#Annotate all the training documents using pyConText
targetPath = site.USER_SITE + "/eHostess/PyConTExtInterface/TargetsAndModifiers/targets.tsv"


targets = itemData.instantiateFromCSVtoitemData(targetPath)
print "Splitting Sentences..."
pyConTextInput = SpanSplitter.splitSentencesMultipleDocuments(notesPath, targets, 10, 10)
pyConText = PyConText.PyConTextInterface()
print "Annotating with PyConText..."
pyConTextDocs = pyConText.PerformAnnotation(pyConTextInput)


# Write the results to the output file
with open("../output/PyConTextOutput_training_101117.txt", 'w') as outFile:
    header = convertAnnotationToCSVString(None, None, None, None)
    outFile.write(header)

    negDocsWithNoAnnotations = 0
    negDocsWithAnnotations = 0
    posDocsWithAnnotations = 0
    posDocsWithoutAnnotations = 0

    for document in pyConTextDocs:
        classification = 0
        if document.documentName in positiveDocs:
            classification = 1
        if classification == 0 and len(document.annotations) == 0:
            negDocsWithNoAnnotations += 1
        elif classification == 0 and len(document.annotations) != 0:
            negDocsWithAnnotations += 1
        elif classification == 1 and len(document.annotations) == 0:
            posDocsWithoutAnnotations += 1
        elif classification == 1 and len(document.annotations) != 0:
            posDocsWithAnnotations += 1
        # if len(document.annotations) == 0:
        #    outFile.write("%s\t%i\n" % (document.documentName, classification))
        # for annotation in document.annotations:
        #     string = convertAnnotationToCSVString(annotation, document.numberOfCharacters, document.documentName, classification)
        #     outFile.write(string)
    print "Negative documents with no annotations: %i" % negDocsWithNoAnnotations
    print "Negative documents with annotations: %i" % negDocsWithAnnotations
    print "Positive documents with no annotations: %i" % posDocsWithoutAnnotations
    print "Positive documents with annotations: %i" % posDocsWithAnnotations

def getDocLength(name):
    with open(notesPath + name + '.txt', 'rU') as inFile:
        return len(inFile.read())

jsonDocs = []
for document in pyConTextDocs:
    dObj = {}
    dObj["name"] = document.documentName
    dObj["trueDocumentClass"] = 0
    if cleanName(document.documentName) in positiveDocs:
        dObj["trueDocumentClass"] = 1
    if document.numberOfCharacters == 0:
        document.numberOfCharacters = getDocLength(document.documentName)
    dObj["documentLength"] = document.numberOfCharacters
    annotations = []
    for annotation in document.annotations:
        aObj = {}
        aObj["text"] =annotation.text
        aObj["predictedMentionClass"] = 1
        if annotation.annotationClass in ["bleeding_absent", "bleeding_historical", "bleeding_hypothetical"]:
            aObj["predictedMentionClass"] = 0
        aObj["target"] = annotation.dynamicProperties["target"]
        aObj["modifiers"] = annotation.dynamicProperties["modifiers"]
        aObj["annotationStart"] = annotation.start
        aObj["annotationEnd"] = annotation.end

        annotations.append(aObj)

    dObj["annotations"] = annotations
    jsonDocs.append(dObj)

jsonString = json.dumps(jsonDocs)
with open("../output/PyConTextOutput_training_101117.json", 'w') as outFile:
    outFile.write(jsonString)



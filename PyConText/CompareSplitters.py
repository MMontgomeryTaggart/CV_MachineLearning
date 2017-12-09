"""This script will accept a list of filepaths and annotates those notes using a chosen sentence splitter."""

import eHostess.PyConTextInterface.PyConText as pyConText
import eHostess.PyConTextInterface.SentenceSplitters.TargetSpanSplitter as SpanSplitter
import eHostess.PyConTextInterface.SentenceSplitters.SpacySplitter as SpacySplitter
import pyConTextNLP.itemData as itemData
import site
import numpy as np
import pandas as pd
import glob
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_fscore_support

TARGETS_PATH = site.USER_SITE + "/eHostess/PyConTExtInterface/TargetsAndModifiers/targets.tsv"
NOTES_DIR = "/Users/shah/Box Sync/MIMC_v2/Corpus_TrainTest/"
#NOTES_DIR = "/Users/shah/Desktop/NoteSubSet/"
GOLD_STANDARD_FILEPATH = "/Users/shah/Box Sync/MIMC_v2/Gold Standard/DocumentClasses.txt"


def getNotesAndClasses(truthPath, balanceClasses=False, invertClasses=False):
    truthData = pd.read_csv(truthPath, dtype={"notes": np.str, "classes": np.int}, delimiter='\t',
                            header=None).as_matrix()

    noteNames = truthData[:, 0].astype(str)
    noteClasses = truthData[:, 1]

    return noteNames, noteClasses


def produceClassifications(docs, names, truth):
    if len(docs) != len(truth):
        raise RuntimeError("There are %i documents but %i truth values." % (len(docs), len(truth)))

    predictions = np.zeros(len(docs))
    for doc in docs:
        docName = doc.documentName
        index = np.argwhere(names == docName)[0]
        foundPositive = False
        for annotation in doc.annotations:
            if annotation.annotationClass == "bleeding_present":
                foundPositive = True
                predictions[index] = 1
        if not foundPositive:
            predictions[index] = 0

    return np.array(truth), predictions


def specificity_score(true, predicted):
    numTrueNegatives = np.sum(np.where(true == 0, 1., 0.))
    numAgreedNegatives = np.sum(((predicted - 1) * -1) * ((true - 1) * -1))
    return float(numAgreedNegatives) / float(numTrueNegatives)


def NPV(true, predicted):
    true = (true - 1) * -1
    predicted = (predicted - 1) * -1
    return precision_score(true, predicted)


def printScores(truth, predictions):
    accuracy = accuracy_score(truth, predictions)
    precision, recall, fscore, support = precision_recall_fscore_support(truth, predictions, average="binary")
    npv = NPV(truth, predictions)
    specificity = specificity_score(truth, predictions)

    print"Accuracy: %.3f\nF-Score: %.3f\nPrecision: %.3f\nRecall (Sensitivity): %.3f\nSpecificity: %.3f\nNPV: %.3f" \
    % (accuracy, fscore, precision, recall, specificity, npv)

targets = itemData.instantiateFromCSVtoitemData(TARGETS_PATH)
sentences = SpanSplitter.splitSentencesMultipleDocuments(NOTES_DIR, targets, 10, 10)
splitter = "Span"
# notesList = glob.glob(NOTES_DIR + "*")
# sentences = SpacySplitter.splitSentencesMultipleDocuments(notesList)
# splitter = "Spacy"
pyConText = pyConText.PyConTextInterface()
docs = pyConText.PerformAnnotation(sentences)

names, classes = getNotesAndClasses(GOLD_STANDARD_FILEPATH)

truth, predictions = produceClassifications(docs, names, classes)

print "Scores for splitter: %s" % (splitter)
printScores(truth, predictions)
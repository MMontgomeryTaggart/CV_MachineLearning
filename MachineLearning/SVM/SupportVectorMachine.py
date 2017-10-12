# Notes: Use k-fold cross validation. sklearn provides nice options for stratifying the data i.e. ensuring that all the
# folds have similar ratios of positive to negative labels, and options for shuffling the data before creating folds,
# which is useful if the data are sorted by class, as they are with us.

# Parameters to vary:
# 1. SVM kernel type and parameters (gamma, coef0, and degree, depending on which kernel you use)
# 2. C, the error term
# 3. The number of features
# 4. The use of bigrams, trigrams, unigrams, stopwords, etc.

# Steps:
# 1. Feature selection
# 2. Training feature extraction
# 3. Model Fitting
# 4. Test feature extraction
# 5. Model evaluation

# POssible modifications:
# 1. change tokenizer to remove words that contain numbers

from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import FeatureSelection
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import pandas as pd
import re

def getNotesAndClasses(corpusPath, truthPath):
    truthData = pd.read_csv(truthPath, dtype={"notes": np.str, "classes": np.int}, delimiter='\t',
                            header=None).as_matrix()

    noteNames = truthData[:, 0].astype(str)
    noteClasses = truthData[:, 1]

    noteBodies = []

    for name in noteNames:
        with open(corpusPath + name + ".txt") as inFile:
            noteBodies.append(inFile.read())
    return np.array(noteBodies), noteClasses.astype(int)

# Get training notes and classes

# Test docs vvvvvvv
corpusPath = "/users/shah/Developer/ShahNLP/MachineLearning/DummyNotes/"
truthDataPath = "/users/shah/Developer/ShahNLP/MachineLearning/DummyNotes/Classes.txt"

corpusPath = "/users/shah/Box Sync/MIMC_v2/Corpus_TrainTest/"
truthDataPath = "/users/shah/Box Sync/MIMC_v2/Gold Standard/DocumentClasses.txt"

noteBodies, noteClasses = getNotesAndClasses(corpusPath, truthDataPath)

# # Get test notes and classes
# testNotesPath = "/users/shah/Developer/ShahNLP/MachineLearning/TestNotes/"
# testNoteClassesPath = "/users/shah/Developer/ShahNLP/MachineLearning/TestNotes/TestClasses.txt"
#
# testNoteBodies, testNoteClasses = getNotesAndClasses(testNotesPath, testNoteClassesPath)

folds = 10
nGramRange = (1, 2)
maxDf = .7
minDf = 5
kFeatures = 1000
C = .1
kernelParam = "rbf"
gammaParam = "auto"
stopWords = None

stratifiedKFold = StratifiedKFold(n_splits=folds)
averageResults = [0., 0., 0., 0., 0.]
count = 0
indicesCombinations = []
for trainingIndices, testingIndices in stratifiedKFold.split(noteBodies, noteClasses):
    indicesCombinations.append((trainingIndices, testingIndices))
for combination in indicesCombinations:
    trainingIndices = combination[0]
    testingIndices = combination[1]

    count +=1
    print "Fold %i" % count
    # print "Training:"
    # print trainingIndices
    # print noteClasses[trainingIndices]
    # print "Testing:"
    # print testingIndices
    # print noteClasses[testingIndices]

    #Generate the raw feature matrix and vocabulary
    extraction = FeatureSelection.FeatureExtraction(nGramRange, kFeatures, maxDf=maxDf, minDf=minDf, stopWords=stopWords, reportTopFeatures=True)
    trainingFeatures, trainingClasses, trainingVocab = extraction.extractFeatures(noteBodies[trainingIndices], noteClasses[trainingIndices])


    # print tfidfVectorizerTrain.get_feature_names()
    # print tfidfMatrixTrain.toarray()
    # print tfidfVectorizerTrain.vocabulary_


    ############ Fit the SVM model ##################

    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier

    model = SVC(C=C, kernel=kernelParam, gamma=gammaParam)
    model = RandomForestClassifier(n_jobs=2, n_estimators=25)
    model.fit(trainingFeatures, trainingClasses)


    ############# Test on test notes #################

    testFeatures, testClasses, vocab = extraction.extractFeatures(noteBodies[testingIndices], noteClasses[testingIndices], vocab=trainingVocab)

    predictions = model.predict(testFeatures)

    precision, recall, fscore, support = precision_recall_fscore_support(noteClasses[testingIndices], predictions, average="binary")
    averageResults[0] += precision
    averageResults[1] += recall
    averageResults[2] += fscore
    # calculate specificity:
    numTrueNegatives = np.sum(np.where(noteClasses[testingIndices] == 0, 1., 0.))
    numAgreedNegatives = np.sum(((predictions - 1) * -1) * ((noteClasses[testingIndices] - 1) * -1))
    specificity = float(numAgreedNegatives) / float(numTrueNegatives)
    averageResults[3] += specificity
    accuracy = float(np.sum(np.where(noteClasses[testingIndices] == predictions, 1, 0))) / float(len(predictions))
    averageResults[4] += accuracy

print "Average results: Precision (PPV): %.4f, Sensitivity (Recall): %.4f, F-Score: %.4f, Specificity: %.4f, Accuracy: %.4f" % tuple(map(lambda x: x/float(folds), averageResults))

with open("./SVMResults.txt", 'a') as outFile:
    outString = "Average results: Precision: %.4f, Sensitivity: %.4f, F-Score: %.4f, Specificity: %.4f, Accuracy: %.4f" % tuple(map(lambda x: x/float(folds), averageResults))
    outString += "     Model: %s, max_df: %.3f, min_df: %.3f, gamma: %s, C: %.3f, kFeatures: %s, stopWords: %s" \
                 % (type(model), maxDf, minDf, str(gammaParam), C, str(kFeatures), stopWords)
    outFile.write(outString + "\n")
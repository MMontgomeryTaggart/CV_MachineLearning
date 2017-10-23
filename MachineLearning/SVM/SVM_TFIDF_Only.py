

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import make_scorer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_recall_fscore_support
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.pipeline import Pipeline
import sklearn.metrics
import numpy as np
import pandas as pd
import re

def selectTopKFeatures(fullFeatures, featureNames, labels, k):
    selector = SelectKBest(chi2, k=k)
    selectedFeatures = selector.fit_transform(fullFeatures, labels)
    selectedIndices = selector.get_support()
    return selectedFeatures, np.array(featureNames)[selectedIndices]

def specificity(true, predicted):
    numTrueNegatives = np.sum(np.where(true == 0, 1., 0.))
    numAgreedNegatives = np.sum(((predicted - 1) * -1) * ((true - 1) * -1))
    return float(numAgreedNegatives) / float(numTrueNegatives)

def NPV(true, predicted):
    true = (true - 1) * -1
    predicted = (predicted - 1) * -1
    return sklearn.metrics.precision_score(true, predicted)

def truePositives(true, predicted):
    return np.sum(np.where(true * predicted == 1, 1, 0))

def trueNegatives(true, predicted):
    true = true - 1
    predicted = predicted - 1
    return np.sum(np.where(true * predicted == 1, 1, 0))

def falseNegatives(true, predicted):
    predicted = (predicted - 1) * -1
    return np.sum(np.where(true * predicted == 1, 1, 0))

def tokenizer(doc):
    tokenPattern = re.compile(r"\b[a-zA-Z]+\b")
    tokens = [token for token in tokenPattern.findall(doc)]
    return tokens


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
if __name__ == "__main__":
    corpusPath = "/users/shah/Box Sync/MIMC_v2/Corpus_TrainTest/"
    truthDataPath = "/users/shah/Box Sync/MIMC_v2/Gold Standard/DocumentClasses.txt"

    noteBodies, labels = getNotesAndClasses(corpusPath, truthDataPath)

    # vectorizer = TfidfVectorizer(ngram_range=(1, 2), tokenizer=tokenizer, sublinear_tf=False)
    # features = vectorizer.fit_transform(noteBodies)

    svm = SVC()
    pipeline = Pipeline([
        ("vectorize", TfidfVectorizer(tokenizer=tokenizer, sublinear_tf=False)),
        ("feature_selection", SelectKBest()),
        ("estimation", svm)
    ])

    parameters = {"vectorize__ngram_range" : [(1, 1), (1, 2), (1, 3)],
                  "vectorize__min_df" : [0.001, 0.01, 0.1],
                  "vectorize__max_df" : [.5, .75, .9, .99],
                  "vectorize__stop_words" : ["english", None],
                  "feature_selection__k" : [50, 100, 500, "all"],
                  "estimation__C" : [100., 1000., 3000., 5000.],
                  "estimation__kernel" : ["linear"]}

    #parameters = {"feature_selection__k" : [100], "estimation__C" : [5000.], "estimation__kernel" : ["linear"]}
    scores = {"Sensitivity" : "recall",
              "Specificity" : make_scorer(specificity),
              "PPV" : "precision",
              "NPV" : make_scorer(NPV),
              "Accuracy" : "accuracy",
              "TruePos" : make_scorer(truePositives)}
    refitScore = "Sensitivity"
    model = GridSearchCV(estimator=pipeline, param_grid=parameters, n_jobs=-1, scoring=scores, refit=refitScore, cv=10, verbose=5)
    model.fit(noteBodies, labels)

    print "Using %s" % type(svm)
    print "Optimized for %s" % refitScore
    print model.best_params_
    bestIndex = model.best_index_
    print "Sensitivity: %.4f" % model.cv_results_["mean_test_Sensitivity"][bestIndex]
    print "Specificity: %.4f" % model.cv_results_["mean_test_Specificity"][bestIndex]
    print "PPV: %.4f" % model.cv_results_["mean_test_PPV"][bestIndex]
    print "NPV: %.4f" % model.cv_results_["mean_test_NPV"][bestIndex]
    print "Accuracy: %.4f" % model.cv_results_["mean_test_Accuracy"][bestIndex]

    predicted = cross_val_predict(model.best_estimator_, noteBodies, labels, cv=10, n_jobs=-1)
    print "params:"
    print model.best_estimator_.get_params()
    print "precision, recall, etc:"
    print precision_recall_fscore_support(labels, predicted, average="binary")
    print "specificity: %.4f" % specificity(labels, predicted)
    print confusion_matrix(labels, predicted)

    # # Get contingency info:

    # bestK = model.best_params_["feature_selection__k"]
    # bestC = model.best_params_["estimation__C"]
    # print ""
    # print "Best C:"
    # print bestC
    # print "Best K:"
    # print bestK
    #
    #
    # selector = SelectKBest(k=100)
    # bestSvm = SVC(C=7000., kernel="linear")
    #
    # selectedFeatures = selector.fit_transform(features, labels)
    #
    # predicted = cross_val_predict(bestSvm, features, labels, cv=10, n_jobs=-1)
    #
    # print "precision, recall, etc:"
    # print precision_recall_fscore_support(labels, predicted, average="binary")
    # print "specificity: %.4f" % specificity(labels, predicted)
    # print confusion_matrix(labels, predicted)


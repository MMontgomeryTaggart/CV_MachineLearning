from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import sklearn
import numpy as np

def specificity(true, predicted):
    numTrueNegatives = np.sum(np.where(true == 0, 1., 0.))
    numAgreedNegatives = np.sum(((predicted - 1) * -1) * ((true - 1) * -1))
    return float(numAgreedNegatives) / float(numTrueNegatives)

def NPV(true, predicted):
    true = (true - 1) * -1
    predicted = (predicted - 1) * -1
    return sklearn.metrics.precision_score(true, predicted)

def printScores(true, predicted):
    precision, recall, fscore, _ = precision_recall_fscore_support(true, predicted, average="binary")
    accuracy = accuracy_score(true, predicted)
    specificity_score = specificity(true, predicted)
    npv = NPV(true, predicted)

    print("Accuracy: %.3f\nF-Score: %.3f\nPrecision: %.3f\nRecall (Sensitivity): %.3f\nSpecificity: %.3f\nNPV: %.3f" % (accuracy, fscore, precision, recall, specificity_score, npv))
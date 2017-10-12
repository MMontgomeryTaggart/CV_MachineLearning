"""This script is intended to take the information produced by pyConText during its mention-level annotation process
and build a document-level classifier."""

import FeatureExtraction
import Scores
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

features, classes = FeatureExtraction.extractPyConTextFeatures("/Users/shah/Box Sync/MIMC_v2/Annotation/Automated/PyConText_ML/PyConTextOutput/PyConTextOutput_training.json")
#random forest params
searchParams = {"n_estimators" : [10, 25, 35, 45], "max_depth" : [1, 3, 5, 10, None], "min_samples_split" : [2, 5, 10, 15, 20, 25]}

#Bagging params
#searchParams = {"n_estimators" : [10, 25, 35, 45], "max_samples" : [.1, .3, .5, .7, 1.], "max_features" : [1, 3, 5, 10, 1.0]}
model = RandomForestClassifier()

refitScorer = "sensitivity"
scores = {"sensitivity" : "recall", "specificity" : make_scorer(Scores.specificity), "precision" : "precision", "F-Score" : "f1", "Accuracy" : "accuracy"}
classifier = GridSearchCV(model, searchParams, cv=10, n_jobs=-1, refit=refitScorer, scoring=scores)
classifier.fit(features, classes)

print "Using %s" % type(model)
print "Optimized for %s" % refitScorer
print classifier.best_params_
bestIndex = classifier.best_index_
print "Sensitivity: %.4f" % classifier.cv_results_["mean_test_sensitivity"][bestIndex]
print "Specificity: %.4f" % classifier.cv_results_["mean_test_specificity"][bestIndex]
print "Precision: %.4f" % classifier.cv_results_["mean_test_precision"][bestIndex]
print "F-Score: %.4f" % classifier.cv_results_["mean_test_F-Score"][bestIndex]
print "Accuracy: %.4f" % classifier.cv_results_["mean_test_Accuracy"][bestIndex]






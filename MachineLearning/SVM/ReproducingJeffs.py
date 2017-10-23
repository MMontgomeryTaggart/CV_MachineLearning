import nltk
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_recall_fscore_support
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
import re

def specificity(true, predicted):
    numTrueNegatives = np.sum(np.where(true == 0, 1., 0.))
    numAgreedNegatives = np.sum(((predicted - 1) * -1) * ((true - 1) * -1))
    return float(numAgreedNegatives) / float(numTrueNegatives)

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


def tokenize(text):
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = [ token for token in tokens if re.search('(^[a-zA-Z]+$)', token) ]
    return filtered_tokens

cachedStopWords = stopwords.words("english") + ['year', 'old', 'man', 'woman', 'ap', 'am', 'pm', 'portable', 'pa', 'lat', 'admitting', 'diagnosis', 'lateral']

# Jeff's
# vectorizer = CountVectorizer(max_df=.8, min_df=.033, ngram_range=(1, 2), preprocessor=None, stop_words=cachedStopWords,
#                              tokenizer=tokenize)
# Mine
vectorizer = CountVectorizer(max_df=.5, min_df=.001, ngram_range=(1, 3), preprocessor=None, stop_words="english",
                             tokenizer=tokenize)
selector = SelectKBest(chi2, k=100)
model = SVC(C=1000., kernel="linear")

corpusPath = "/users/shah/Box Sync/MIMC_v2/Corpus_TrainTest/"
truthDataPath = "/users/shah/Box Sync/MIMC_v2/Gold Standard/DocumentClasses.txt"

print "Reading notes..."
noteBodies, labels = getNotesAndClasses(corpusPath, truthDataPath)

features = vectorizer.fit_transform(noteBodies)

print "predicting..."
prediction = cross_val_predict(model, noteBodies, labels, cv=10, n_jobs=-1, verbose=10)

print precision_recall_fscore_support(labels, prediction, average="binary")

# print "creating pipeline"
# pipeline = Pipeline([("extraction", Tf), ("selection", selector), ("prediction", model)])
# grid = GridSearchCV(pipeline,
#                     scoring={"sensitivity" : "recall", "specificity" : make_scorer(specificity), "precision": "precision", "F_Score" : "f1", "accuracy": "accuracy"},
#                     refit="sensitivity",
#                     param_grid={"extraction__max_df" : [.5],
#                                 "extraction__min_df" : [0.001],
#                                 "extraction__ngram_range": [(1, 3)],
#                                 "extraction__stop_words": ["english"],
#                                 "extraction__tokenizer" : [tokenize],
#                                 "selection__k" : [100],
#                                 "prediction__C" :[1000.],
#                                 "prediction__kernel" : ["linear"]},
#                     cv=10,
#                     n_jobs=-1)
# grid.fit(noteBodies, labels)
#
# print grid.best_params_
# print grid.best_score_
# bestIndex = grid.best_index_
# print "Sensitivity: %.4f" % grid.cv_results_["mean_test_sensitivity"][bestIndex]
# print "Specificity: %.4f" % grid.cv_results_["mean_test_specificity"][bestIndex]
# print "PPV: %.4f" % grid.cv_results_["mean_test_precision"][bestIndex]
# print "Accuracy: %.4f" % grid.cv_results_["mean_test_accuracy"][bestIndex]


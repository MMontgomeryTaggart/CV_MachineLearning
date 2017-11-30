import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from os.path import expanduser
import pandas as pd
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
import re

class TokenTracker(object):
    def __init__(self, targetTokens):
        self.tokenCount = 0
        self.targetTokens = targetTokens

    def track(self, token):

        if token == self.targetTokens[self.tokenCount]:
            self.tokenCount += 1
        else:
            self.tokenCount = 0
        if self.tokenCount == len(self.targetTokens):
            self.tokenCount = 0
            return True
        else:
            return False

tokenTracker = TokenTracker(["bleed", "gi", "bleed"])

def tokenizer(text):
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = [token for token in tokens if re.search('(^[a-zA-Z]+$)', token)]
    a = []
    for i in filtered_tokens:
        i = WordNetLemmatizer().lemmatize(i, 'v')
        a.append(i)
        if tokenTracker.track(i):
            print text
    return a
    #return filtered_tokens


def getNotesAndClasses(corpusPath, truthPath, balanceClasses=False):
    truthData = pd.read_csv(truthPath, dtype={"notes": np.str, "classes": np.int}, delimiter='\t',
                            header=None).as_matrix()

    noteNames = truthData[:, 0].astype(str)
    noteClasses = truthData[:, 1]

    if balanceClasses:
        np.random.seed(8229)
        noteNames = np.array(noteNames)
        posIndices = np.where(noteClasses == 1)[0]
        negIndices = np.where(noteClasses == 0)[0]
        posNotes = noteNames[posIndices]
        negNotes = noteNames[negIndices]
        assert len(posNotes) + len(negNotes) == len(noteNames)

        selectedNegNotes = np.random.choice(negNotes, size=len(posNotes), replace=False)
        allNotes = np.concatenate((posNotes, selectedNegNotes), axis=0)
        labels = np.concatenate((np.ones(len(posNotes)), np.zeros(len(selectedNegNotes))), axis=0)

        noteNames = allNotes
        noteClasses = labels

    noteBodies = []

    for name in noteNames:
        with open(corpusPath + name + ".txt") as inFile:
            noteBodies.append(inFile.read())
    return np.array(noteBodies), noteClasses.astype(int)
if __name__ == "__main__":
    homeDir = expanduser("~")
    corpusPath = homeDir + "/Box Sync/MIMC_v2/Corpus_TrainTest/"
    truthDataPath = homeDir + "/Box Sync/MIMC_v2/Gold Standard/DocumentClasses.txt"

    noteBodies, labels = getNotesAndClasses(corpusPath, truthDataPath, balanceClasses=False)

    vectorizer = TfidfVectorizer(ngram_range=(1, 3), min_df=0.001, max_df=.5, stop_words="english", tokenizer=tokenizer)

    transformedFeatures = vectorizer.fit_transform(noteBodies)

    selector = SelectKBest(chi2, k=100)
    selector.fit(transformedFeatures, labels)

    scores = selector.scores_
    featureNames = vectorizer.get_feature_names()
    pairs = zip(featureNames, scores)
    pairs.sort(key=lambda pair: pair[1], reverse=True)

    for index in range(20):
        print "%s: %.3f" % (pairs[index][0], pairs[index][1])
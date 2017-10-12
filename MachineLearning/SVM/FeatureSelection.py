from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import numpy as np
import re
import nltk

def tokenize(text):
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = [ token for token in tokens if re.search('(^[a-zA-Z]+$)', token) ]
    return filtered_tokens

def filterFeatures(vocabularyList, matrix):
    #remove digits:
    pattern = re.compile(r"\b\d+\b")
    nonDigitColumns = [index for index, word in enumerate(vocabularyList) if not pattern.match(word)]

    cleanMatrix = matrix[:, nonDigitColumns]
    cleanVocab = [vocabularyList[index] for index in nonDigitColumns]

    return cleanVocab, cleanMatrix



class FeatureExtraction(object):
    def __init__(self, nGramRange, kFeatures, maxDf=None, minDf=None, stopWords=None, reportTopFeatures=False):
        self.nGramRange = nGramRange
        self.kFeatures = kFeatures
        self.maxDf = maxDf
        self.minDf = minDf
        self.stopWords = stopWords
        self.reportTopFeatures = reportTopFeatures

    def extractNGrams(self, notes, classes, vocab):
        vectorizer = TfidfVectorizer(ngram_range=self.nGramRange, max_df=self.maxDf, min_df=self.minDf, stop_words=self.stopWords,
                                     tokenizer=tokenize)
        if type(vocab) != type(None):
            vectorizer = TfidfVectorizer(vocabulary=vocab)
            vectorizedNotes = vectorizer.fit_transform(notes)
            return vocab, vectorizedNotes
        vectorizedNotes = vectorizer.fit_transform(notes)
        fullVocab = vectorizer.get_feature_names()
        # Clean out digit features
        cleanVocab, cleanVectors = filterFeatures(fullVocab, vectorizedNotes)

        selector = SelectKBest(chi2, k=self.kFeatures)
        selector.fit(cleanVectors, classes)
        selectedFeatureMask = selector.get_support(indices=True)
        cleanVocab = np.array(cleanVocab)

        return cleanVocab[selectedFeatureMask], cleanVectors[:, selectedFeatureMask]

    def extractCustomFeatures(self, notes):
        # Custom features: 1. "(?<!non)(?<!non )(?<!non-)(?<!no )bleed" count normalized down column
        # 2. "(?<!non)(?<!non )(?<!non-)(?<!no )bleed" count > 6
        # 3. "(?<!non)(?<!non )(?<!non-)(?<!no )bleed" count > 2
        # 3. "blood\s+loss" abs count > 2 (implies bleeding absent)
        # 4. "(?<!non-)(?<!non)(?<!non )bloody" abs count > 1
        # 5. "(?<!non)(?<!non )(?<!non-)(?<!no )bleed" abs count normalized across the column
        # 6. "hem{1,2}or{1,2}h{1,2}age?" abs count > 3 (or maybe 4, 3 seems like an overcommitment)
        # 7. "((\bg|gua?iac)([\-]|\s+)((pos(itive)?)|\+)|guaiac\(\+\))" abs count normalized across column
        # 8. "coffee[\-\s]+(ground|grounds)" abs count normalized across column
        # 9 "(?<!no\s)(?<!non\s)mel[ae]n(a|ic)" > 4
        # 10. "brbpr" count normalized down column

        numFeatures = 10
        features = np.zeros((len(notes), numFeatures))

        for index, note in enumerate(notes):
            vec = np.zeros(10)
            vec[0] = len(re.findall("(?<!non)(?<!non )(?<!non-)(?<!no )bleed", note, re.I))
            vec[1] = 0
            if len(re.findall("(?<!non)(?<!non )(?<!non-)(?<!no )bleed", note, re.I)) > 6:
                vec[1] = 1
            vec[2] = 0
            if len(re.findall("(?<!non)(?<!non )(?<!non-)(?<!no )bleed", note, re.I)) > 2:
                vec[2] = 1
            vec[3] = 0
            if len(re.findall("blood\s+loss", note, re.I)) > 2:
                vec[3] = 1
            vec[4] = 0
            if len(re.findall("(?<!non-)(?<!non)(?<!non )bloody", note, re.I)) > 1:
                vec[4] = 1
            vec[5] = 0
            if len(re.findall("hem{1,2}or{1,2}h{1,2}age?", note, re.I)) > 3:
                vec[5] = 1
            vec[6] = len(re.findall("((\bg|gua?iac)([\-]|\s+)((pos(itive)?)|\+)|guaiac\(\+\))", note, re.I))
            vec[7] = len(re.findall("coffee[\-\s]+(ground|grounds)", note, re.I))
            vec[8] = 0
            if len(re.findall("(?<!no\s)(?<!non\s)mel[ae]n(a|ic)", note, re.I)) > 4:
                vec[8] = 1
            vec[9] = len(re.findall("brbpr", note, re.I))
            features[index] = vec
        normalizer = np.ones(numFeatures)
        normalizer[0] = np.max(features[:, 0])
        normalizer[6] = np.max(features[:, 6])
        normalizer[7] = np.max(features[:, 7])
        normalizer[9] = np.max(features[:, 9])
        for index, value in enumerate(normalizer):
            if value == 0:
                normalizer[index] = 1.

        features = features / normalizer

        return features





    def extractFeatures(self, notes, classes, vocab=None):
        """Returns a list of n-grams to use as features when generating the training and test data."""

        nGramsVocab, nGramsVectors = self.extractNGrams(notes, classes, vocab=vocab)
        customFeatures = self.extractCustomFeatures(notes)

        combinedFeatures = np.append(nGramsVectors.toarray(), customFeatures, axis=1)
        #selector =
        # selector.fit(combinedFeatures, classes)
        # selectedFeatureMask = selector.get_support(indices=True)
        # cleanVocab = np.array(cleanVocab)
        # selectedFeatures = cleanVocab[selectedFeatureMask]
        # selectedPValues = selector.pvalues_[selectedFeatureMask]
        # selectedScores = selector.scores_[selectedFeatureMask]
        #
        # if self.reportTopFeatures:
        #     # report the top k features
        #     triplets = zip(selectedFeatures, selectedPValues, selectedScores)
        #     print "Selected Triplets:"
        #     print sorted(triplets, key=lambda x: x[1])
        #
        # vectorizer = TfidfVectorizer(vocabulary=selectedFeatures)
        #
        # trainingVectors = vectorizer.fit_transform(notes)

        return combinedFeatures, classes, nGramsVocab



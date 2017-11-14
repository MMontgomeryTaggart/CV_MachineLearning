from gensim.models.phrases import Phrases
from gensim.models.phrases import Phraser
import glob
import os
import sys
import pickle
import numpy as np
import spacy
import re
import codecs


nlp = spacy.load("en", disable=["ner"])

CORPUS_PATH = "/Users/shah/Box Sync/MIMC_v2/Corpus/corpus/corpus/*"


class DocumentGenerator(object):
    def __init__(self, corpusDir, randomSubset=False):
        self.corpusDir = corpusDir
        self.randomSubsample = randomSubset
        np.random.seed(10)

    def __iter__(self):
        fileList = glob.glob(self.corpusDir)
        if self.randomSubsample:
            fileList = np.random.choice(fileList, self.randomSubsample)
        print("Found {} files.".format(len(fileList)))
        numFiles = len(fileList)
        for index, filename in enumerate(fileList):
            sys.stdout.write("\rProcessing {0} of {1}. ({2:.3f}%)".format(index + 1, numFiles, (float(index + 1) / float(numFiles)) * 100))
            sys.stdout.flush()
            if os.path.isfile(filename):
                with codecs.open(filename, 'r', encoding='utf-8') as f:
                    body = f.read()
                body = re.sub(r"\[\*\*.+\*\*\]", "", body)
                body = re.sub(r"\n|\r", " ", body)
                yield body


        print()

class ProcessedDocGenerator(object):
    def __init__(self, docBodyGenerator):
        self.docBodyGenerator = docBodyGenerator

    def __iter__(self):
        for doc in nlp.pipe(self.docBodyGenerator, batch_size=500, n_threads=4):
            for sent in doc.sents:
                tokens = [token.lemma_ for token in sent if not re.match("\s+", token.text)]
                yield tokens


phrases = Phrases(ProcessedDocGenerator(DocumentGenerator(CORPUS_PATH)))
phrases.save("SerializedObjects/Phrases.obj")
#phrasesObject = pickle.load(open("SerializedObjects/Phrases.pkl", 'rb'))
# for phrase, score in phrases.export_phrases(ProcessedDocGenerator(DocumentGenerator(CORPUS_PATH, randomSubset=50))):
#     print(u'{0}   {1}'.format(phrase, score))
bigrammer = Phraser(phrases)
trigramPhrases = Phrases(bigrammer[ProcessedDocGenerator(DocumentGenerator(CORPUS_PATH))])
trigramPhrases.save("SerializedObjects/TrigramPhrases.obj")

print(bigrammer[["the", "patient", "past", "medical", "history"]])

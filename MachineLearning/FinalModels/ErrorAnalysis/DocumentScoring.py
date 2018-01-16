import pickle
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import nltk
from nltk import WordNetLemmatizer
import collections


TRAINING_CORPUS_PATH = "/Users/shah/Box Sync/MIMC_v2/Corpus_TrainTest/"
TRAINING_TRUTH_PATH = "/Users/shah/Box Sync/MIMC_v2/Gold Standard/DocumentClasses.txt"
TEST_CORPUS_PATH = "/Users/shah/Developer/ShahNLP/TestNotes/Notes/"
TEST_TRUTH_PATH = "/Users/shah/Developer/ShahNLP/TestNotes/TestDocumentClasses.txt"

TRAIN_FRAME_PATH = "./Predictions/PredictionFrames/TrainFrame.pkl"
TRAIN_DS_FRAME_PATH = "./Predictions/PredictionFrames/TrainDSFrame.pkl"
TEST_FRAME_PATH = "./Predictions/PredictionFrames/TestFrame.pkl"
TEST_DS_FRAME_PATH = "./Predictions/PredictionFrames/TestDSFrame.pkl"

SVM_MODEL_PATH = "../SerializedModels/SVMNotDownsampledFinal.pkl"
SVM_DS_MODEL_PATH = "../SerializedModels/SVMFinal.pkl"

NoteInfo = collections.namedtuple("NoteInfo", ["name", "group", "customScore", "modelScore"])

def tokenizer(text):
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = [token for token in tokens if re.search('(^[a-zA-Z]+$)', token)]
    a = []
    for i in filtered_tokens:
        a.append(WordNetLemmatizer().lemmatize(i, 'v'))
    return a

def calculateVocabPosNegRatio(vocab, notes, classes, dev=False):
    if dev:
        zeros = np.zeros(len(notes))
        tuples = zip(vocab, zeros)
        return dict(tuples)

    ratios = {}
    for token in vocab:
        posCount = 0
        negCount = 0
        for index, note in enumerate(notes):
            matches = len(re.findall(re.escape(token), note, re.I))
            if classes[index] == 1:
                posCount += matches
            else:
                negCount += matches
        currentRatio = (float(posCount) + 1.) / (float(negCount) + 1.)
        ratios[token] = currentRatio
    return ratios

def getModelWeights(modelPath, vocab):
    pipeline = pickle.load(open(modelPath))
    model = pipeline.named_steps["estimation"]
    coefficients = np.squeeze(np.asarray(model.coef_.todense()))
    weightDict = {}
    for index, token in enumerate(vocab):
        weightDict[token] = coefficients[index]
    return weightDict

def determineNoteGroup(vocab, notes, noteNames, classes, dev=False):
    # classes:
    # 0: bleeding negative, no vocab words
    # 1: bleeding negative, with vocab words
    # 2: bleeding positive, no vocab words
    # 3: bleeding positive, with vocab words

    if dev:
        zeros = np.zeros(len(notes))
        tuples = zip(noteNames, zeros)
        return dict(tuples)

    counts = [0, 0, 0, 0]

    groups = {}
    for index, note in enumerate(notes):
        noteName = noteNames[index]
        foundVocab = False
        for token in vocab:
            if re.search(re.escape(token), note, re.I):
                foundVocab = True
                break
        if foundVocab:
            if classes[index] == 1:
                # positive note, with vocab
                groups[noteName] = 3
                counts[3] += 1
            else:
                # negative note with vocab
                groups[noteName] = 1
                counts[1] += 1
        else:
            if classes[index] == 1:
                # positive note, no vocab
                groups[noteName] = 2
                counts[2] += 1
            else:
                # negative note, no vocab
                groups[noteName] = 0
                counts[0] += 1
    print("Groups Counts: %s" % str(counts))
    return groups


def determineSortingScore(ratios, notes, noteNames, dev=False):
    if dev:
        zeros = np.zeros(len(notes))
        tuples = zip(noteNames, zeros)
        return dict(tuples)

    sortingScores = {}
    for index, note in enumerate(notes):
        noteName = noteNames[index]
        noteScore = 1.
        for token, ratio in ratios.items():
            numTokens = len(re.findall(re.escape(token), note, re.I))
            scoreAdjustment = float(numTokens) * ratio
            noteScore += scoreAdjustment
        sortingScores[noteName] = noteScore
    return sortingScores

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
    #convert notes to tokenized notes for string matching later, since the features were generated from tokenized notes
    tokenizedLists = list(map(tokenizer, noteBodies))
    tokenizedNotes = map(lambda noteList: " ".join(noteList), tokenizedLists)
    return np.asarray(tokenizedNotes), noteNames, noteClasses.astype(int)

def getVocab(notes, classes, k=100):
    # # Best Params
    # parameters = {"vectorize__ngram_range": [(1, 3)],
    #               "vectorize__min_df": [0.001],
    #               "vectorize__max_df": [.5],
    #               "vectorize__stop_words": ["english"],
    vectorizer = TfidfVectorizer(ngram_range=(1,3), min_df=0.001, max_df=.5, stop_words="english")
    vectorizedNotes = vectorizer.fit_transform(notes)
    features = np.asarray(vectorizer.get_feature_names())
    selector = SelectKBest(chi2, k=k)
    selector.fit(vectorizedNotes, classes)
    selectedIndices = selector.get_support()

    return features[selectedIndices]

def getGroupsAndSortingScores(notes, noteNames, classes, modelPath, dev=False):
    noteTuples = []
    vocab = getVocab(notes, classes)
    groups = determineNoteGroup(vocab, notes, noteNames, classes, dev=dev)
    ratios = calculateVocabPosNegRatio(vocab, notes, classes, dev=dev)
    modelWeights = getModelWeights(modelPath, vocab)
    customSortingScores = determineSortingScore(ratios, notes, noteNames, dev=dev)
    modelSortingScores = determineSortingScore(modelWeights, notes, noteNames, dev=dev)

    assert len(groups.keys()) == len(customSortingScores.keys()) and len(groups.keys()) == len(notes)

    for noteName in groups.keys():
        noteTuple = NoteInfo(name=noteName, group=groups[noteName], customScore=customSortingScores[noteName], modelScore=modelSortingScores[noteName])
        noteTuples.append(noteTuple)
    return noteTuples, ratios

def insertInfoIntoFrame(tuples, frame):
    frameNames = frame["names"]
    tuples.sort(key=lambda tuple: frameNames[frameNames == tuple.name].index[0])
    noteInfoDict = {"names": [], "group" : [], "customSortScore" : [], "modelSortScore": []}
    for noteTuple in tuples:
        noteInfoDict["names"].append(noteTuple.name)
        noteInfoDict["group"].append(noteTuple.group)
        noteInfoDict["customSortScore"].append(noteTuple.customScore)
        noteInfoDict["modelSortScore"].append(noteTuple.modelScore)
    noteInfoFrame = pd.DataFrame(noteInfoDict)
    mergedFrame = pd.merge(frame, noteInfoFrame, on="names")
    return mergedFrame

TrainFrame = pickle.load(open(TRAIN_FRAME_PATH))
TrainDSFrame = pickle.load(open(TRAIN_DS_FRAME_PATH))
TestFrame = pickle.load(open(TEST_FRAME_PATH))
TestDSFrame = pickle.load(open(TEST_DS_FRAME_PATH))

trainingNotes, trainingNoteNames, trainingNoteClasses = getNotesAndClasses(TRAINING_CORPUS_PATH, TRAINING_TRUTH_PATH)
trainingDSNotes, trainingDSNoteNames, trainingDSNoteClasses = getNotesAndClasses(TRAINING_CORPUS_PATH, TRAINING_TRUTH_PATH, balanceClasses=True)
testNotes, testNoteNames, testNoteClasses = getNotesAndClasses(TEST_CORPUS_PATH, TEST_TRUTH_PATH)

dev = False

print("Training:")
trainingNoteTuples, trainingVocabRatios = getGroupsAndSortingScores(trainingNotes, trainingNoteNames, trainingNoteClasses, SVM_MODEL_PATH, dev=dev)
TrainFrameWithScores = insertInfoIntoFrame(trainingNoteTuples, TrainFrame)
print("")

print("TrainingDS:")
trainingDSNoteTuples, trainingDSVocabRatios = getGroupsAndSortingScores(trainingDSNotes, trainingDSNoteNames, trainingDSNoteClasses, SVM_DS_MODEL_PATH, dev=dev)
TrainDSFrameWithScores = insertInfoIntoFrame(trainingDSNoteTuples, TrainDSFrame)
print("")

print("Test:")
testNoteTuples, testVocabRatios = getGroupsAndSortingScores(testNotes, testNoteNames, testNoteClasses, SVM_MODEL_PATH, dev=dev)
TestFrameWithScores = insertInfoIntoFrame(testNoteTuples, TestFrame)
print("")
print("Test DS:")
testDSNoteTuples, testDSVocabRatios = getGroupsAndSortingScores(testNotes, testNoteNames, testNoteClasses, SVM_DS_MODEL_PATH, dev=dev)
TestDSFrameWithScores = insertInfoIntoFrame(testDSNoteTuples, TestDSFrame)
print("")


pickle.dump(TrainFrameWithScores, open("./Predictions/PredictionFrames/TrainFrameWithScores.pkl", "wb"))
pickle.dump(TrainDSFrameWithScores, open("./Predictions/PredictionFrames/TrainDSFrameWithScores.pkl", "wb"))
pickle.dump(TestFrameWithScores, open("./Predictions/PredictionFrames/TestFrameWithScores.pkl", "wb"))
pickle.dump(TestDSFrameWithScores, open("./Predictions/PredictionFrames/TestDSFrameWithScores.pkl", "wb"))

print(TrainFrameWithScores)
print(TrainDSFrameWithScores)
print(TestFrameWithScores)
print(TestDSFrameWithScores)



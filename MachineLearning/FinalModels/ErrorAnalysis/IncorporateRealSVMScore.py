import numpy as np
import pandas as pd
import pickle
import re
import nltk
from nltk import WordNetLemmatizer

TRAINING_PATH = "/Users/shah/Box Sync/MIMC_v2/Corpus_TrainTest/"
TRAINING_TRUTH_PATH = "/Users/shah/Box Sync/MIMC_v2/Gold Standard/DocumentClasses.txt"
CORPUS_PATH = "/Users/shah/Developer/ShahNLP/TestNotes/Notes/"
TRUTH_PATH = "/Users/shah/Developer/ShahNLP/TestNotes/TestDocumentClasses.txt"

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
    return np.array(noteBodies), noteClasses.astype(int), noteNames

def tokenizer(text):
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = [token for token in tokens if re.search('(^[a-zA-Z]+$)', token)]
    a = []
    for i in filtered_tokens:
        a.append(WordNetLemmatizer().lemmatize(i, 'v'))
    return a
    #return filtered_tokens

training_notes, training_labels, training_names = getNotesAndClasses(TRAINING_PATH, TRAINING_TRUTH_PATH, balanceClasses=False)
test_notes, test_labels, test_names = getNotesAndClasses(CORPUS_PATH, TRUTH_PATH, balanceClasses=False)

# For the non-downsampled notes
training_model = pickle.load(open("../SerializedModels/SVMNotDownsampledFinal.pkl"))

vectorizer = training_model.named_steps["vectorize"]
selector = training_model.named_steps["feature_selection"]
estimator = training_model.named_steps["estimation"]

training_notes_vectorized = vectorizer.transform(training_notes)
training_notes_selected = np.asarray(selector.transform(training_notes_vectorized).todense())

test_notes_vectorized = vectorizer.transform(test_notes)
test_notes_selected = np.asarray(selector.transform(test_notes_vectorized).todense())

coefs = np.squeeze(np.asarray(estimator.coef_.todense()))
intercept = estimator.intercept_[0]

raw_training_scores = np.matmul(coefs, training_notes_selected.T) + intercept
raw_test_scores = np.matmul(coefs, test_notes_selected.T) + intercept

data = {"names" : [], "RealSVMScore" : []}

data["names"].extend(training_names)
data["names"].extend(test_names)

data["RealSVMScore"].extend(raw_training_scores)
data["RealSVMScore"].extend(raw_test_scores)

score_frame = pd.DataFrame(data)

# For downsampled notes
training_model_ds = pickle.load(open("../SerializedModels/SVMFinal.pkl"))

vectorizer_ds = training_model_ds.named_steps["vectorize"]
selector_ds = training_model_ds.named_steps["feature_selection"]
estimator_ds = training_model_ds.named_steps["estimation"]

training_notes_vectorized_ds = vectorizer_ds.transform(training_notes)
training_notes_selected_ds = np.asarray(selector_ds.transform(training_notes_vectorized_ds).todense())

test_notes_vectorized_ds = vectorizer_ds.transform(test_notes)
test_notes_selected_ds = np.asarray(selector_ds.transform(test_notes_vectorized_ds).todense())

coefs_ds = np.squeeze(np.asarray(estimator_ds.coef_.todense()))
intercept_ds = estimator_ds.intercept_[0]

raw_training_scores_ds = np.matmul(coefs_ds, training_notes_selected_ds.T) + intercept_ds
# raw_test_scores_ds = np.matmul(coefs_ds, test_notes_selected_ds.T) + intercept_ds
#
# data_ds = {"names" : [], "RealSVMScore" : []}
#
# data_ds["names"].extend(training_names)
# data_ds["names"].extend(test_names)
#
# data_ds["RealSVMScore"].extend(raw_training_scores_ds)
# data_ds["RealSVMScore"].extend(raw_test_scores_ds)
#
# score_frame_ds = pd.DataFrame(data_ds)
#
#
#
# TrainFrameWithScores = pickle.load(open("./Predictions/PredictionFrames/TrainFrameWithScoresWithPyConText.pkl", "rb"))
# TrainDSFrameWithScores = pickle.load(open("./Predictions/PredictionFrames/TrainDSFrameWithScores.pkl", "rb"))
# TestFrameWithScores = pickle.load(open("./Predictions/PredictionFrames/TestFrameWithScoresWithPyConText.pkl", "rb"))
# TestDSFrameWithScores = pickle.load(open("./Predictions/PredictionFrames/TestDSFrameWithScores.pkl", "rb"))
#
# frames = [TrainFrameWithScores, TrainDSFrameWithScores, TestFrameWithScores, TestDSFrameWithScores]
#
# TrainFrameWithScores = pd.merge(TrainFrameWithScores, score_frame, on="names", how="inner")
# TrainDSFrameWithScores = pd.merge(TrainDSFrameWithScores, score_frame_ds, on="names", how="inner")
# TestFrameWithScores = pd.merge(TestFrameWithScores, score_frame, on="names", how="inner")
# TestDSFrameWithScores = pd.merge(TestDSFrameWithScores, score_frame_ds, on="names", how="inner")
#
#
# pickle.dump(TrainFrameWithScores, open("./Predictions/PredictionFrames/TrainFrameWithSVM.pkl", "wb"))
# pickle.dump(TrainDSFrameWithScores, open("./Predictions/PredictionFrames/TrainDSFrameWithSVM.pkl", "wb"))
# pickle.dump(TestFrameWithScores, open("./Predictions/PredictionFrames/TestFrameWithSVM.pkl", "wb"))
# pickle.dump(TestDSFrameWithScores, open("./Predictions/PredictionFrames/TestDSFrameWithSVM.pkl", "wb"))

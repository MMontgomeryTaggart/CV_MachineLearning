import os
import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import keras
import pickle
from Scoring import printScores

TRAINING_PATH = "/Users/shah/Box Sync/MIMC_v2/Corpus_TrainTest/"
TRAINING_TRUTH_PATH = "/Users/shah/Box Sync/MIMC_v2/Gold Standard/DocumentClasses.txt"

MAX_NUM_WORDS = 2000
TEST_CORPUS_PATH = "/Users/shah/Developer/ShahNLP/TestNotes/Notes/"
TEST_TRUTH_PATH = "/Users/shah/Developer/ShahNLP/TestNotes/TestDocumentClasses.txt"
MODEL_PATH = "./SerializedModels/CNNNotDownsampledFinal.h5"
GOLD_STANDARD_PATH = "/Users/shah/Box Sync/MIMC_v2/Gold Standard/DocumentClasses.txt"
CORPUS_TRAIN_PATH = "/Users/shah/Box Sync/MIMC_v2/Corpus_TrainTest/"

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
            noteBodies.append(inFile.read().lower())
    return np.array(noteBodies), noteClasses.astype(int)


myPath = os.path.dirname(os.path.realpath(__file__))
# baseDir = "./TensorBoardLogs/"
# runName = "Batch.8.Epochs.12"
# os.makedirs(baseDir + runName,)

cleanTexts, labels = getNotesAndClasses(CORPUS_TRAIN_PATH, GOLD_STANDARD_PATH, balanceClasses=False)
trainTexts, trainLabels = getNotesAndClasses(CORPUS_TRAIN_PATH, GOLD_STANDARD_PATH, balanceClasses=False)
#labels = np.where(numberLabels == 1, "Positive", "Negative")
numLabels = 2

# Start fitting the convolutional model
tokenizer = Tokenizer()
tokenizer.fit_on_texts(trainTexts)
sequences = tokenizer.texts_to_sequences(cleanTexts)

wordIndex = tokenizer.word_index
print("Found %i words." % len(wordIndex))

x_test = pad_sequences(sequences, maxlen=MAX_NUM_WORDS)
#labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', x_test.shape)
print('Shape of label tensor:', labels.shape)


model = keras.models.load_model(MODEL_PATH)

rawPredictions = model.predict(x_test)
predictions = np.reshape(np.where(rawPredictions > .5, 1, 0), (rawPredictions.shape[0],))

pickle.dump(predictions, open("./ErrorAnalysis/Predictions/CNN_Training_Predictions.pkl", 'wb'))

printScores(labels, predictions)

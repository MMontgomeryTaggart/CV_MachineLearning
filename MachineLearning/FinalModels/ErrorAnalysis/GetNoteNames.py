import pickle
import numpy as np
import pandas as pd

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


    return np.array(noteNames), noteClasses.astype(int)


trainingNames, trainingClasses = getNotesAndClasses(TRAINING_PATH, TRAINING_TRUTH_PATH)
trainingDSNames, trainingDSClasses = getNotesAndClasses(TRAINING_PATH, TRAINING_TRUTH_PATH, balanceClasses=True)
testNames, testClasses = getNotesAndClasses(CORPUS_PATH, TRUTH_PATH)

pickle.dump({"names" : trainingNames, "classes" : trainingClasses}, open("./Predictions/TrainingNamesAndClasses.pkl", "wb"))
pickle.dump({"names" : trainingDSNames, "classes" : trainingDSClasses}, open("./Predictions/TrainingDSNamesAndClasses.pkl", "wb"))
pickle.dump({"names" : testNames, "classes" : testClasses}, open("./Predictions/TestNamesAndClasses.pkl", "wb"))
import numpy as np
import pandas as pd

TEST_CORPUS_PATH = "/Users/shah/Developer/ShahNLP/TestNotes/Notes/"
TEST_TRUTH_PATH = "/Users/shah/Developer/ShahNLP/TestNotes/TestDocumentClasses.txt"

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


notes, classes = getNotesAndClasses(TEST_CORPUS_PATH, TEST_TRUTH_PATH)

assert(len(notes) == len(classes))

print("Test Set Statistics:")
print("Number of notes: %i" % len(classes))
print("Num Positive: %i" % np.sum(classes))
print("Num Negative: %i" % (len(classes) - np.sum(classes)))
print("Prevalence: %.3f" % np.mean(classes))
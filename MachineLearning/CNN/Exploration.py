import glob
import numpy as np

CORPUS_TRAIN_PATH = "/Users/shah/Box Sync/MIMC_v2/Corpus_TrainTest/*"

files = glob.glob(CORPUS_TRAIN_PATH)
numWords = np.zeros(len(files))

for index, filename in enumerate(files):
    with open(filename, 'rU') as f:
        note = f.read()

    numWords[index] = float(len(note.split()))

avg = np.mean(numWords)
std = np.std(numWords)
max = np.max(numWords)
min = np.min(numWords)

print "Average: %.2f\nStd: %.2f\nMax: %.1f\nMin: %.1f" % (avg, std, max, min)
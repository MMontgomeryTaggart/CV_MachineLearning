import pandas as pd
import numpy as np
import re

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

def getSplit(regex, positiveNotes, negativeNotes):
    """
    1. number of documents that contain the regex in each group, abs value and percentage
    2. average number of occurances in each group, max number of occurances
    """

    positiveCounts = np.zeros(len(positiveNotes))
    negativeCounts = np.zeros(len(negativeNotes))

    inputTuples = [(positiveNotes, 'positive'), (negativeNotes, 'negative')]
    for pair in inputTuples:
        notes = pair[0]
        noteClass = pair[1]
        for index, note in enumerate(notes):
            numOccurances = len(re.findall(regex, note, re.I))
            if noteClass == 'negative':
                negativeCounts[index] = numOccurances
            else:
                positiveCounts[index] = numOccurances

    positiveCount = np.sum(np.where(positiveCounts != 0, 1, 0)).astype(int)
    negativeCount = np.sum(np.where(negativeCounts != 0, 1, 0)).astype(int)
    print "Positive Count: %i" % positiveCount
    print "Positive Percentage: %.3f" % (float(positiveCount) / len(positiveNotes))
    print "Positive Average (present): %.3f" % np.mean(positiveCounts[np.where(positiveCounts != 0)])
    print "Positive Max: %.1f" % np.max(positiveCounts)
    print "Positive Min (present): %.1f" % np.min(positiveCounts[np.where(positiveCounts != 0)])
    print ""
    print "Negative Count: %i" % negativeCount
    print "Negative Percentage: %.3f" % (float(negativeCount) / len(negativeNotes))
    print "Negative Average (present): %.3f" % np.mean(negativeCounts[np.where(negativeCounts != 0)])
    negativeMax = np.max(negativeCounts)
    print "Negative Max: %.1f" % negativeMax
    print "Negative Min (present): %.1f" % np.min(negativeCounts[np.where(negativeCounts != 0)])
    print ""
    print "More than negative max: %i" % len(np.where(positiveCounts > negativeMax)[0])
    print positiveCounts[np.where(positiveCounts > negativeMax)]



corpusPath = "/users/shah/Box Sync/MIMC_v2/Corpus_TrainTest/"
truthDataPath = "/users/shah/Box Sync/MIMC_v2/Gold Standard/DocumentClasses.txt"

noteBodies, noteClasses = getNotesAndClasses(corpusPath, truthDataPath)

positiveNotes = noteBodies[np.where(noteClasses == 1)]
negativeNotes = noteBodies[np.where(noteClasses == 0)]

getSplit("coffee[\-\s]+(ground|grounds)", positiveNotes, negativeNotes)

# Possible new features

# 1. "(?<!non)(?<!non )(?<!non-)(?<!no )bleed" abs count is > 6:
# Positive Count: 160
# Positive Percentage: 0.711
# Positive Average (present): 3.112
# Positive Max: 20.0
# Positive Min (present): 1.0
#
# More than 6 count: 1
# [ 11.   8.   8.   9.  20.   8.   8.   7.   8.   7.   9.  10.  10.  10.  10.
#    9.  13.]
#
# Negative Count: 138
# Negative Percentage: 0.180
# Negative Average (present): 1.609
# Negative Max: 6.0
# Negative Min (present): 1.0

# 2. "(?<!non)(?<!non )(?<!non-)(?<!no )bleed" abs count is > 2:

# 3. "blood\s+loss" abs count > 2 (implies bleeding absent)

# 4. "(?<!non-)(?<!non)(?<!non )bloody" abs count > 1

# 5. "(?<!non)(?<!non )(?<!non-)(?<!no )bleed" abs count normalized across the column

# 6. "hem{1,2}or{1,2}h{1,2}age?" abs count > 3 (or maybe 4, 3 seems like an overcommitment)

# 7. "((\bg|gua?iac)([\-]|\s+)((pos(itive)?)|\+)|guaiac\(\+\))" abs count normalized across column

# 8. "coffee[\-\s]+(ground|grounds)" abs count normalized across column

# 9 "(?<!no\s)(?<!non\s)mel[ae]n(a|ic)" > 4

# 10. "brbpr" count normalized down column


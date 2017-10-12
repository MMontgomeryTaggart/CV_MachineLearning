import pandas as pd
import glob
import numpy as np
import re

targetFilePath = '/Users/shah/Developer/PythonVirtualEnv/lib/python2.7/site-packages/eHostess/PyConTextInterface/TargetsAndModifiers/targets.tsv'

targetsFrame = pd.read_csv(targetFilePath, sep='\t')

targetNames = targetsFrame["Lex"].as_matrix()
targetRegexes = targetsFrame["Regex"].as_matrix()

docClassPath = "/users/shah/Box Sync/MIMC_v2/Gold Standard/DocumentClasses.txt"
docClassFrame = pd.read_csv(docClassPath, sep='\t', header=None, names=["name", "class"], dtype={"name" : np.str, "class" : np.int64})

# get list of documents in first four batches
corpusDirectories = ['/users/shah/Box Sync/MIMC_v2/Annotation/Adjudication/batch_0/corpus/*',
                     '/users/shah/Box Sync/MIMC_v2/Annotation/Adjudication/batch_1/corpus/*',
                     '/users/shah/Box Sync/MIMC_v2/Annotation/Adjudication/batch_2/corpus/*',
                     '/users/shah/Box Sync/MIMC_v2/Annotation/Adjudication/batch_3/corpus/*']

docNamesRaw = []
for dir in corpusDirectories:
    docNamesRaw.extend(glob.glob(dir))

def cleanNameFunc(name):
    fullName = name.split('/')[-1]
    noExtension = fullName.split('.')[0]
    return noExtension

pilotNames = map(cleanNameFunc, docNamesRaw)

allNames = docClassFrame["name"].as_matrix()
allClasses = docClassFrame["class"].as_matrix()

indices = []
for name in pilotNames:
    index = np.where(allNames == name)[0][0]
    indices.append(index)

pilotNames = allNames[indices]
pilotClasses = allClasses[indices]

positiveFrequencies = np.zeros(len(targetNames))
negativeFrequencies = np.zeros(len(targetNames))

for docPath in docNamesRaw:
    with open(docPath, 'rU') as inFile:
        noteBody = inFile.read()
    cleanName = cleanNameFunc(docPath)
    indexOfName = np.where(pilotNames == cleanName)[0][0]
    docClass = pilotClasses[indexOfName]

    for index, regex in enumerate(targetRegexes):
        count = len(re.findall(regex, noteBody, re.I))
        if docClass == 1:
            positiveFrequencies[index] += count
        else:
            negativeFrequencies[index] += count

totalFrequencies = negativeFrequencies + positiveFrequencies
groups = zip(targetNames, totalFrequencies, positiveFrequencies, negativeFrequencies)
groups.sort(key=lambda x: x[1], reverse=True)
initialLists = map(list, zip(*groups))
targetNames = initialLists[0]
totalFrequencies = initialLists[1]
positiveFrequencies = initialLists[2]
negativeFrequencies = initialLists[3]
with open("Frequencies.txt", 'w') as outFile:
    for name in targetNames:
        outFile.write(name + '\t')
    outFile.write('\n')
    for freq in totalFrequencies:
        outFile.write("%i\t" % freq)
    outFile.write('\n')
    for freq in positiveFrequencies:
        outFile.write("%i\t" % freq)
    outFile.write('\n')
    for freq in negativeFrequencies:
        outFile.write("%i\t" % freq)
    outFile.write('\n')




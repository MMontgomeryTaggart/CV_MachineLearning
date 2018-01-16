import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import confusion_matrix

testFrame = pickle.load(open("./Predictions/PredictionFrames/TestFrameWithSVM.pkl", 'rb'))
dsFrame = pickle.load(open("./Predictions/PredictionFrames/TestDSFrameWithSVM.pkl", 'rb'))

gold = testFrame["gold"].as_matrix()
pyConText = testFrame["PyConText_Test_Predictions"].as_matrix()
et = dsFrame["ET_DS_Test_Predictions"].as_matrix()
notes = testFrame["names"].as_matrix()

pyConTextConfusion = confusion_matrix(gold, pyConText)
etConfusion = confusion_matrix(gold, et)

firstWhere = np.where(np.logical_and(pyConText != gold, et != gold))

overlappingIndices = np.where(np.logical_and(pyConText != gold, et != gold))
overlappingNotes = notes[overlappingIndices]


overlappingFalseNegatives = overlappingNotes[np.where(gold[overlappingIndices] == 1)]
overlappingFalsePositives = overlappingNotes[np.where(gold[overlappingIndices] == 0)]

print pyConTextConfusion
print etConfusion
print overlappingNotes
print len(overlappingNotes)
print overlappingFalseNegatives
print len(overlappingFalseNegatives)
print overlappingFalsePositives
print len(overlappingFalsePositives)

print testFrame[testFrame["names"]=="405930"]
print "**************************"
print dsFrame[dsFrame["names"]=="405930"]

pyConTextFalseNegatives = notes[np.where(np.logical_and(gold == 1, pyConText == 0))]
print len(pyConTextFalseNegatives)
print pyConTextFalseNegatives


combinedFrame = pd.merge(testFrame, dsFrame, on="names")

outPath = "./TempOut.txt"

with open(outPath, 'a') as f:
    f.write("NoteName,Gold,PyConText,ET-DS,Revised,Notes\n")

for index, row in combinedFrame.iterrows():
    with open(outPath, 'a') as f:
        if row['PyConText_Test_Predictions'] != row["gold_x"] or row['ET_DS_Test_Predictions'] != row["gold_x"]:
            f.write("%s,%i,%i,%i\n" % (row["names"], int(row["gold_x"]), int(row['PyConText_Test_Predictions']), int(row['ET_DS_Test_Predictions'])))

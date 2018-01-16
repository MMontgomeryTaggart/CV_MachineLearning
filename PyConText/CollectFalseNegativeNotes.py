import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import confusion_matrix

testFrame = pickle.load(open("../MachineLearning/FinalModels/ErrorAnalysis/Predictions/PredictionFrames/TestFrameWithSVM.pkl", 'rb'))
pyConText = testFrame["PyConText_Test_Predictions"].as_matrix()
notes = testFrame["names"].as_matrix()
gold = testFrame["gold"].as_matrix()

falseNegatives = notes[np.where(np.logical_and(gold == 1, pyConText == 0))]

print falseNegatives
print len(falseNegatives)


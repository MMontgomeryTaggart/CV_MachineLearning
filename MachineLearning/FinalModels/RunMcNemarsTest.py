import numpy as np
import pickle


TestFrameWithSVM = pickle.load(open("./ErrorAnalysis/Predictions/PredictionFrames/TestFrameWithSVM.pkl", "rb"))
TestDSFrameWithSVM = pickle.load(open("./ErrorAnalysis/Predictions/PredictionFrames/TestDSFrameWithSVM.pkl", "rb"))

gold = TestDSFrameWithSVM['gold'].as_matrix()
pyConTextPredictions = TestFrameWithSVM['PyConText_Test_Predictions'].as_matrix()
etDSPredictions = TestDSFrameWithSVM["ET_DS_Test_Predictions"].as_matrix()


pyConTextPositive = pyConTextPredictions[np.where(gold == 1)]
etPositive = etDSPredictions[np.where(gold == 1)]

b = np.sum(np.where(np.logical_and(pyConTextPositive==1, etPositive==0), 1, 0), dtype=np.float)
c = np.sum(np.where(np.logical_and(pyConTextPositive==0, etPositive==1), 1, 0), dtype=np.float)

print b
print c

x2 = ((b - c) ** 2) / (b + c)

print x2
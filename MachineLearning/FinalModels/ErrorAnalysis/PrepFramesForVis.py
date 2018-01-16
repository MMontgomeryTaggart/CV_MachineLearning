import pandas as pd
import numpy as np
import pickle

FRAME_OUTPUT_DIR = "/Users/shah/Developer/ShahNLP/MachineLearning/FinalModels/ErrorAnalysis/Predictions/PredictionFrames/VisFrames/CSV/"

TrainFrameWithSVM = pickle.load(open("./Predictions/PredictionFrames/TrainFrameWithSVM.pkl", "rb"))
TrainDSFrameWithSVM = pickle.load(open("./Predictions/PredictionFrames/TrainDSFrameWithSVM.pkl", "rb"))
TestFrameWithSVM = pickle.load(open("./Predictions/PredictionFrames/TestFrameWithSVM.pkl", "rb"))
TestDSFrameWithSVM = pickle.load(open("./Predictions/PredictionFrames/TestDSFrameWithSVM.pkl", "rb"))

frames = {"Train": TrainFrameWithSVM, "TrainDS" : TrainDSFrameWithSVM, "Test" : TestFrameWithSVM, "TestDS" : TestDSFrameWithSVM}



for name, frame in frames.items():
    frame.to_csv(FRAME_OUTPUT_DIR + name + "withSVM.csv", sep='\t', index=False)


# Old frames:
# TrainFrameWithScores = pickle.load(open("./Predictions/PredictionFrames/TrainFrameWithScoresWithPyConText.pkl", "rb"))
# TrainDSFrameWithScores = pickle.load(open("./Predictions/PredictionFrames/TrainDSFrameWithScores.pkl", "rb"))
# TestFrameWithScores = pickle.load(open("./Predictions/PredictionFrames/TestFrameWithScoresWithPyConText.pkl", "rb"))
# TestDSFrameWithScores = pickle.load(open("./Predictions/PredictionFrames/TestDSFrameWithScores.pkl", "rb"))
#
# frames = {"Train": TrainFrameWithScores, "TrainDS" : TrainDSFrameWithScores, "Test" : TestFrameWithScores, "TestDS" : TestDSFrameWithScores}
#
# for key, frame in frames.items():
#     groupValues = frame["group"].unique()
#     tempFrame = pd.DataFrame()
#     for groupNum in groupValues:
#         groupRows = frame.loc[frame["group"] == groupNum]
#         groupRows = groupRows.sort_values("modelSortScore")
#         groupRows["modelSortOrder"] = range(len(groupRows))
#         groupRows = groupRows.sort_values("customSortScore")
#         groupRows["customSortOrder"] = range(len(groupRows))
#         groupRows = groupRows[["names", "modelSortOrder", "customSortOrder"]]
#         tempFrame = tempFrame.append(groupRows)
#         #print tempFrame
#     frame = pd.merge(frame, tempFrame, on="names")
#     print key
#     print frame
#     pickle.dump(frame, open("/Users/shah/Developer/ShahNLP/MachineLearning/FinalModels/ErrorAnalysis/Predictions/PredictionFrames/VisFrames/" + key + "VisFrame.pkl", 'wb'))
#     frame.to_csv("/Users/shah/Developer/ShahNLP/MachineLearning/FinalModels/ErrorAnalysis/Predictions/PredictionFrames/VisFrames/CSV/" + key + "Data.csv", sep='\t', index=False)



import pickle
import pandas as pd
import matplotlib.pyplot as plt

TrainFrameWithScores = pickle.load(open("./Predictions/PredictionFrames/TrainFrameWithSVM.pkl", "rb"))
TrainDSFrameWithScores = pickle.load(open("./Predictions/PredictionFrames/TrainDSFrameWithSVM.pkl", "rb"))
TestFrameWithScores = pickle.load(open("./Predictions/PredictionFrames/TestFrameWithSVM.pkl", "rb"))
TestDSFrameWithScores = pickle.load(open("./Predictions/PredictionFrames/TestDSFrameWithSVM.pkl", "rb"))

frames = [TrainFrameWithScores, TrainDSFrameWithScores, TestFrameWithScores, TestDSFrameWithScores]

for index, frame in enumerate(frames):
    print frame
    #pd.DataFrame.hist(frame, column="RealSVMScore", bins=5000)
    #plt.show()
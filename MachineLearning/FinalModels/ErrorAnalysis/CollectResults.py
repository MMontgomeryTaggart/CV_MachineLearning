import pandas as pd
import pickle

modelsList = ["CNN", "ET", "SVM"]
samplingList = ["", "DS"]
groupList = ["Training", "Test"]

TrainDSPredictions = {}
TrainPredictions = {}
TestDSPredictions = {}
TestPredictions = {}
for sampleType in samplingList:
    for group in groupList:
        for model in modelsList:
                tempSample = sampleType
                if sampleType != "":
                    tempSample = sampleType + "_"
                id = model + "_" + tempSample + group + "_Predictions"
                filename = id + ".pkl"
                predictions = pickle.load(open("./Predictions/" + filename, 'rb'))
                if sampleType == "DS":
                    if group == "Training":
                        TrainDSPredictions[id] = predictions
                    else:
                        TestDSPredictions[id] = predictions
                else:
                    if group == "Training":
                        TrainPredictions[id] = predictions
                    else:
                        TestPredictions[id] = predictions

#Get names and classes
trainingNamesClasses = pickle.load(open("./Predictions/TrainingNamesAndClasses.pkl", "rb"))
trainingDSNamesClasses = pickle.load(open("./Predictions/TrainingDSNamesAndClasses.pkl", "rb"))
testNamesClasses = pickle.load(open("./Predictions/TestNamesAndClasses.pkl", "rb"))

TrainDSPredictions["names"] = trainingDSNamesClasses["names"]
TrainDSPredictions["gold"] = trainingDSNamesClasses["classes"]

TrainPredictions["names"] = trainingNamesClasses["names"]
TrainPredictions["gold"] = trainingNamesClasses["classes"]

TestDSPredictions["names"] = testNamesClasses["names"]
TestDSPredictions["gold"] = testNamesClasses["classes"]

TestPredictions["names"] = testNamesClasses["names"]
TestPredictions["gold"] = testNamesClasses["classes"]

TrainDSFrame = pd.DataFrame(TrainDSPredictions)
TrainFrame = pd.DataFrame(TrainPredictions)
TestDSFrame = pd.DataFrame(TestDSPredictions)
TestFrame = pd.DataFrame(TestPredictions)

# Serialize Prediction Frames
pickle.dump(TrainDSFrame, open("./Predictions/PredictionFrames/TrainDSFrame.pkl", 'wb'), 2)
pickle.dump(TrainFrame, open("./Predictions/PredictionFrames/TrainFrame.pkl", 'wb'), 2)
pickle.dump(TestDSFrame, open("./Predictions/PredictionFrames/TestDSFrame.pkl", 'wb'), 2)
pickle.dump(TestFrame, open("./Predictions/PredictionFrames/TestFrame.pkl", 'wb'), 2)




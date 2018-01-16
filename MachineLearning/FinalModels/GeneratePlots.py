import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

TRAINING_DATA_PATH = "./ResultsOnTrainingSet.tsv"
TEST_DATA_PATH = "./ResultsOnTestSet.tsv"

def matColor(r, g, b):
    denom = 255.
    return tuple(map(lambda x: float(x) / denom, [r, g, b]))

def plotLabels(data, xName, yName, xOffset=0.015, yOffset=0.01):
    for row in data.itertuples():
        name = row.Model
        xValue = row.__getattribute__(xName)
        yValue = row.__getattribute__(yName)
        plt.text(xValue + xOffset, yValue + yOffset, name)

def plotData(data, title, xName, yName, xLabel, yLabel):
    sns.set_style("whitegrid")
    plt.figure()
    plt.plot(data[xName], data[yName], ms=10.0, mfc=matColor(232, 152, 12),
             mec=matColor(232, 152, 12), marker="o", color="None")
    plt.axis([0, 1, 0, 1])
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(title)


trainingData = pd.read_csv(TRAINING_DATA_PATH, delimiter='\t')
testData = pd.read_csv(TEST_DATA_PATH, delimiter='\t')

# plotData(trainingData, "Training Results", "PPV", "Sensitivity", "Positive Predictive Value", "Sensitivity")
# plotLabels(trainingData, "PPV", "Sensitivity")

plotData(testData, "Test Results", "PPV", "Sensitivity", "Positive Predictive Value", "Sensitivity")
plotLabels(testData, "PPV", "Sensitivity")

plotData(testData, "Test Results - NPVSpecificity", "NPV", "Specificity", "Negative Predictive Value", "Specificity")
plotLabels(testData, "NPV", "Specificity")
plt.savefig("/users/shah/Desktop/ResultPlots/Test Results - NPVSpecificity.svg")

plt.show()

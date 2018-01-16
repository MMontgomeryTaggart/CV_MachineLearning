import pandas as pd
import numpy as np

TRAINING_PATH = "./Output/PatientData_Training.csv"
TEST_PATH = "./Output/PatientData_Test.csv"
trainingFrame = pd.read_csv(TRAINING_PATH, delimiter='\t')
testFrame = pd.read_csv(TEST_PATH, delimiter='\t')


def printFrameStatistics(frame, name):
    print(name + ":")
    totalPatients = len(frame["subject_id"].unique())

    print("Total Patients: %i" % totalPatients)

    #Determine sex and age
    nonDuplicates = frame.drop_duplicates(subset="subject_id")
    numMale = len(nonDuplicates[nonDuplicates["gender"]=="M"])
    numFemale = len(nonDuplicates[nonDuplicates["gender"] == "F"])
    assert (numMale + numFemale) == totalPatients

    print("Male Patients: %i (%.2f%%)" % (numMale, float(numMale) * 100. / float(totalPatients)))
    print("Female Patients: %i (%.2f%%)" % (numFemale, float(numFemale) * 100. / float(totalPatients)))

    malePatients = nonDuplicates[nonDuplicates["gender"]=="M"]
    femalePatients = nonDuplicates[nonDuplicates["gender"] == "F"]

    numDeadMales = len(malePatients[malePatients["death"]!='None'])
    numDeadFemales = len(femalePatients[femalePatients["death"] != 'None'])

    print("Num deceased males: %i (%.2f%%)" % (numDeadMales, float(numDeadMales) * 100. / float(numMale)))
    print("Num deceased females: %i (%.2f%%)" % (numDeadFemales, float(numDeadFemales) * 100. / float(numFemale)))
    print 'hi'


printFrameStatistics(testFrame, "Test Demographics")
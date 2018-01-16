import MySQLdb
import pandas as pd
import numpy as np



TRAINING_PATH = "/Users/shah/Box Sync/MIMC_v2/Corpus_TrainTest/"
TRAINING_TRUTH_PATH = "/Users/shah/Box Sync/MIMC_v2/Gold Standard/DocumentClasses.txt"
TEST_PATH = "/Users/shah/Developer/ShahNLP/TestNotes/Notes/"
TEST_TRUTH_PATH = "/Users/shah/Developer/ShahNLP/TestNotes/TestDocumentClasses.txt"

def getNotesAndClasses(corpusPath, truthPath, balanceClasses=False):
    truthData = pd.read_csv(truthPath, dtype={"notes": np.str, "classes": np.int}, delimiter='\t',
                            header=None).as_matrix()

    noteNames = truthData[:, 0].astype(str)
    noteClasses = truthData[:, 1]

    if balanceClasses:
        np.random.seed(8229)
        noteNames = np.array(noteNames)
        posIndices = np.where(noteClasses == 1)[0]
        negIndices = np.where(noteClasses == 0)[0]
        posNotes = noteNames[posIndices]
        negNotes = noteNames[negIndices]
        assert len(posNotes) + len(negNotes) == len(noteNames)

        selectedNegNotes = np.random.choice(negNotes, size=len(posNotes), replace=False)
        allNotes = np.concatenate((posNotes, selectedNegNotes), axis=0)
        labels = np.concatenate((np.ones(len(posNotes)), np.zeros(len(selectedNegNotes))), axis=0)

        noteNames = allNotes
        noteClasses = labels

    noteBodies = []

    for name in noteNames:
        with open(corpusPath + name + ".txt") as inFile:
            noteBodies.append(inFile.read())
    return np.array(noteBodies), noteClasses.astype(int), noteNames

def getData(noteNameList, outPath):
    columns = ["subject_id", "note_name", "dob", "death", "gender", "seq_num", "icd9_code"]

    db = MySQLdb.Connect(
        host="mysql.chpc.utah.edu",
        user="mimicro",
        passwd="M5rltpg0D",
        db="mimic3"
    )

    c = db.cursor()

    nameString = ",".join(noteNameList)

    query = """SELECT p.SUBJECT_ID, n.ROW_ID, p.DOB, p.DOD, p.GENDER, d.SEQ_NUM, d.ICD9_CODE FROM PATIENTS AS p INNER JOIN NOTEEVENTS AS n ON n.SUBJECT_ID=p.SUBJECT_ID INNER JOIN DIAGNOSES_ICD AS d ON d.SUBJECT_ID=p.SUBJECT_ID WHERE d.SEQ_NUM=1 AND n.ROW_ID IN (%s)""" % nameString
    #query = """SELECT p.SUBJECT_ID, n.ROW_ID, p.DOB, p.GENDER FROM PATIENTS AS p INNER JOIN NOTEEVENTS AS n ON n.SUBJECT_ID=p.SUBJECT_ID  WHERE n.ROW_ID IN (%s)""" % nameString

    c.execute(query)

    with open(outPath, 'w') as f:
        f.write('\t'.join(columns) + "\n")
        for row in c.fetchall():
            f.write('\t'.join(map(str, row)) + "\n")


_, _, trainingNoteNames = getNotesAndClasses(TRAINING_PATH, TRAINING_TRUTH_PATH)
_, _, testNoteNames = getNotesAndClasses(TEST_PATH, TEST_TRUTH_PATH)

outPath = "./Output/PatientData_Training.csv"
getData(trainingNoteNames, outPath)
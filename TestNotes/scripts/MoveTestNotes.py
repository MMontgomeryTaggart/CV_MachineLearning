import shutil
import glob

DESTINATION_DIR = "../Notes/"

batchRange = range(33, 54 + 1)

paths = []
for batchNo in batchRange:
    notesDir = "/Users/shah/Developer/ShahNLP/TestNotes/batches/batch_%i/corpus/*" % batchNo
    notePaths = glob.glob(notesDir)
    paths.extend(notePaths)

for path in paths:
    shutil.copy(path, DESTINATION_DIR)


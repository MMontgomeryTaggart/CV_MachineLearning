"""Reads-in all the document bodies and stores them in a dictionary keyed by document name."""

import glob
import os

def parseTextDocs(textsDirectorypath):
    if not os.path.exists(textsDirectorypath):
        raise ValueError("Path does not exist: %s" % textsDirectorypath)
    if textsDirectorypath[-1] != '*':
        if textsDirectorypath[-1] != '/':
            textsDirectorypath += '/*'
        else:
            textsDirectorypath += '*'
    fileList = glob.glob(textsDirectorypath)

    texts = {}

    for file in fileList:
        fileName = file.split("/")[-1]
        with open(file, 'r') as inFile:
            texts[fileName] = inFile.read()

    return texts
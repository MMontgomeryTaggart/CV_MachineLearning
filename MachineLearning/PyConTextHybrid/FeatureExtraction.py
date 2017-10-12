"""This file is intended to contain classes to extract features from the JSON data produced by pyConText during its
annotation process."""

import json
import numpy as np
import re

def mentionLevelClassIsPositive(annotation):
    if annotation["predictedMentionClass"] == 1:
        return 1.
    else:
        return 0.

def getTargetAndModifierCounts(document, numFeatures):
    targeRegexList = [
        r"(?<!non)(?<!non )(?<!non-)bleed[a-z]*",
        r"blood loss",
        r"blood per rectum",
        r"(?<!non-)(?<!non)(?<!non )bloody",
        r"brbpr",
        r"coffee[\-\s](ground|grounds)",
        r"ecchymos[ie]s",
        r"epistaxis",
        r"exsanguination",
        r"backward",
        r"((\bg|gua?iac)([\-]|\s+)((pos(itive)?)|\+)|guaiac\(\+\))",
        r"hematem[a-z]+",
        r"hematochezia",
        r"hematoma",
        r"hematuria",
        r"hemoperitoneum",
        r"hemoptysis",
        r"hem{1,2}or{1,2}h{1,2}age?",
        r"\bich",
        r"mel[ae]n(a|ic)",
        r"(ng|ngt)\s+lavage\s+((positive)|(pos)|\+)",
        r"((positive)|(pos)|\+) (ng|ngt) lavage",
        r"(fecal\s+occult(\s+blood)?|\bob|\bfob)\s+pos(itive)?",
        r"sah",
        r"sdh",
        r"(maroon|red)\s+(stool|bowel\s+movement|bm)",
        r"vomit[a-z]* blood"
    ]
    modifierRegexList = [
        r"\bago\b",
        r"(cc:|chief complaint:)",
        r"ddx",
        r"denies|denied",
        r"did not did not (show|reveal)",
        r"episode of",
        r"episodes of",
        r"found to found to have",
        r"here here (with|w\\|w/)",
        r"((h/o)|(h\\o)|(hx of)|history)",
        r"history history\s+of",
        r"\bif\b",
        r"in the \bin\s+the\s+past/b",
        r"monitor( for)?",
        r"\bnegative\b",
        r"\bnever\b",
        r"\bno\b",
        r"no no (evidence|e\\o|e/o)",
        r"no no ((h/o)|(h\\o)]|hx)",
        r"non(\s|-)",
        r"not\b",
        r"now now (with|w\\|w/)",
        r"possible",
        r"presenting* (with|w\\|w/)",
        r"present[s|e][d]* (with|w\\|w/)",
        r"previous",
        r"\bprior\b",
        r"risk of",
        r"rule (rule out|r\/o)",
        r"recent recent\s+admission",
        r"suspicion",
        r"transfuse",
        r"unlikely",
        r"versus|vs",
        r"watch for",
        r"((without)|(w\\o)|(w/o))(?!\scontrast)"
    ]

    counts = np.zeros(numFeatures - 2)
    # Targets
    for index, regexString in enumerate(targeRegexList):
        for annotation in document["annotations"]:
            if re.search(regexString, annotation["target"], re.I):
                counts[index] += 1.
                continue
    for index, regexString in enumerate(modifierRegexList):
        for annotation in document["annotations"]:
            for modifier in annotation["modifiers"]:
                if re.search(regexString, modifier, re.I):
                    counts[index + len(targeRegexList)] += 1.
                    continue
    return counts

def extractPyConTextFeatures(jsonPath):
    with open(jsonPath, 'rU') as inFile:
        documentList = json.load(inFile)

    # Features:
    # Number of positive annotations
    # Number of negative annotations
    # One feature each for number of particular modifiers and targets
    numFeatures = 2 + 36 + 27
    features = np.zeros((len(documentList), numFeatures))
    classes = np.zeros(len(documentList))
    for index, document in enumerate(documentList):
        classes[index] = float(document["trueDocumentClass"])
        vec = np.zeros(numFeatures)
        # num positive features:
        numPositive = np.sum(list(map(mentionLevelClassIsPositive, document["annotations"])))
        vec[0] = numPositive
        # num negative features
        numNegative = len(document["annotations"]) - numPositive
        vec[1] = numNegative
        vec[2:] = getTargetAndModifierCounts(document, numFeatures)

        features[index] = vec

    return features, classes


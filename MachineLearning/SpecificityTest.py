import numpy as np
from sklearn.metrics import precision_recall_fscore_support

true = np.array([1, 1, 1, 0, 0, 0])
predicted = np.array([1, 1, 1, 1, 1, 0])

numTrueNegatives = np.sum(np.where(true == 0, 1., 0.))
numAgreedNegatives = np.sum(((predicted - 1) * -1) * ((true - 1) * -1))
specificity = float(numAgreedNegatives) / float(numTrueNegatives)

print specificity
print precision_recall_fscore_support(true, predicted, average="binary")
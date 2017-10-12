import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

sampleDocs = ["John has a black cat",
              "Francis has a red cat",
              "Milly has a white cat",
              "John has a black dog"]

countVectorizer = CountVectorizer(token_pattern=u"(?u)\\b\\w+\\b")
countVectorizer.fit(sampleDocs)

countMatrix = countVectorizer.transform(sampleDocs)

print "Vocabulary:"
print countVectorizer.get_feature_names()
print ""
print "Word-count matrix:"
print countMatrix.toarray()

transformer = TfidfTransformer(smooth_idf=False)
transformer.fit(countMatrix)

print "Normalized from numpy.TfidfTransformer:"
tfidfMatrix = transformer.transform(countMatrix)
print tfidfMatrix.toarray()
print ""
print ""


# I do it myself!
countMatrixCopy = countMatrix.toarray()
numDocuments = 4.
totalFrequencies = np.sum(countMatrixCopy, axis=0, dtype=np.float64)
idf = np.log(numDocuments / totalFrequencies) + 1

myTfidfMatrix = countMatrixCopy * idf

rowSums = np.power(np.sum(np.power(myTfidfMatrix, [[2.]]), axis=1, keepdims=True), [[.5]])
myTfidfMatrixNormed = myTfidfMatrix / rowSums

print "Normalized from self-implementation:"
print myTfidfMatrixNormed
print ""
print ""


print "Un-normalized from numpy.TfidfTransformer:"
transformer = TfidfTransformer(smooth_idf=False, norm=False)
transformer.fit(countMatrix)

tfidfMatrix = transformer.transform(countMatrix)
print tfidfMatrix.toarray()
print ""
print ""


print "Un-normalized from self-implementation:"
print myTfidfMatrix
print ""
print ""
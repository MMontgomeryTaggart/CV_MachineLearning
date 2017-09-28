import eHostess.eHostInterface.KnowtatorReader as Reader
import eHostess.PyConTextInterface.PyConText as PyConText
import eHostess.PyConTextInterface.SentenceSplitters.TargetSpanSplitter as SpanSplitter
from eHostess.Analysis.DocumentComparison import Comparison
from eHostess.Analysis.Output import ConvertComparisonsToTSV
from eHostess.Analysis.Metrics import CalculateRecallPrecisionFScoreAndAgreement
import pyConTextNLP.itemData as itemData
import site

# knowtatorPath = "/users/shah/Box Sync/MIMC_v2/Corpus/corpus/TrainingNotes/saved/"
# notesPath = "/users/shah/Box Sync/MIMC_v2/Corpus/corpus/TrainingNotes/corpus/"

knowtatorPath = "/users/shah/Developer/testNotes/saved/"
notesPath = "/users/shah/Developer/testNotes/corpus/"

targetPath = site.USER_SITE + "/eHostess/PyConTExtInterface/TargetsAndModifiers/targets.tsv"

reader = Reader.KnowtatorReader()
print "Creating documents from the knowtator files..."
humanDocs = reader.parseMultipleKnowtatorFiles(knowtatorPath)

targets = itemData.instantiateFromCSVtoitemData(targetPath)
print "Splitting Sentences..."
pyConTextInput = SpanSplitter.splitSentencesMultipleDocuments(notesPath, targets, 10, 10)
pyConText = PyConText.PyConTextInterface()
print "Annotating with PyConText..."
pyConTextDocs = pyConText.PerformAnnotation(pyConTextInput)


comparisons = Comparison.CompareDocumentBatches(humanDocs,
                                                pyConTextDocs,
                                                equivalentClasses=[["bleeding_present"], ["bleeding_absent", "bleeding_hypthetical", "bleeding_historical"]],
                                                equivalentAttributes=False,
                                                countNoOverlapAsMatch=["bleeding_absent", "bleeding_hypothetical", "bleeding_historical"])

ConvertComparisonsToTSV(comparisons, "./output/PyConTextTrainingDocumentAnnotationResults.txt")

print CalculateRecallPrecisionFScoreAndAgreement(comparisons)

import re
import nltk
import numpy as np
import pandas as pd
import pickle as pkl
from sklearn.model_selection import KFold
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from nltk.stem.wordnet import WordNetLemmatizer
from Scoring import printScores

TRAINING_PATH = ""
CORPUS_PATH = "/Users/shah/Developer/ShahNLP/TestNotes/Notes/"
TRUTH_PATH = "/Users/shah/Developer/ShahNLP/TestNotes/TestDocumentClasses.txt"

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
    return np.array(noteBodies), noteClasses.astype(int)



def tokenize(text):
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = [ token for token in tokens if re.search('(^[a-zA-Z]+$)', token) ]
    a=[]
    for i in filtered_tokens:
        a.append(WordNetLemmatizer().lemmatize(i,'v'))
    return a
    #return filtered_tokens


cachedStopWords = stopwords.words("english") + ['age','ago','also','already',
                                                'x','year', 'old', 'man', 'woman', 'ap',
                                                'am', 'pm', 'portable', 'pa', 'lat',
                                                'admitting', 'diagnosis', 'lateral',
                                               'bb','bp','c','daily','data','date','abd','abg',
                                               'mg','ast','av','ck','cm','cr','cv','cvp','cpk','cx','day','dp','ed','f','ffp'
                                               ,'hct','hd','icu','ii','id','ml','af','arf','bs',
                                               'cc','ccu','hr','ef','fen','hpi','l','k','r','ra','abx'
                                               'alk','phos','iv','ext','gi','iv','ivf','ni','ng','vs','vt','yo','yn',
                                               'zosyn','kg','abx','alk','alt','ckmb','ct','cta','p','pe','po','c','ck','ca'
                                               'q','cr','ni','ett','iv','g','h','j','k','l','z','x','c','v','b','n','m','i','ii',
                                               'iii','iv','kg','lll','lvh','mb','mcg','md','ml','xl','wnl','wgt',
                                                'q','w','e','r','t','y','u','i','o','p','first','gm','hcl','hs','hrs',
                                               'inr','mmm','mr','mri','mrsa','ms','lf','nl','ns','nsr','sh','nt','tf','tr'
                                               ,'wbc','plt','bcx','bph','bmp','mmhg','bps','sq','ld','ce','cbc','ckd',
                                               'cp','cxr','cva','cvicu','dm','dr','name','ep','er','gtt','iabp','cxr',
                                               'jvd','jvp','pt','kvo','lbs','na','nad','nd','nph','npo','osh',
                                               ]


corpusList, labels = getNotesAndClasses(CORPUS_PATH, TRUTH_PATH)
vocab = pkl.load(open("./SerializedModels/ExtraTreesVocabularyNotDownsampledEnglishStopOnly.pkl", "rb"))
print("Number of vocab terms: %i" % len(vocab))

### before one

cv = TfidfVectorizer(lowercase=True,
                     ngram_range=(1, 3), preprocessor=None, stop_words=cachedStopWords,
                     strip_accents=None, tokenizer=tokenize, vocabulary=vocab)
X = cv.fit_transform(corpusList)
print(X.shape)
print()
lexicon = cv.get_feature_names()
#print (lexicon)
print()


Y = np.array(labels)
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

print(X.shape)

#X = SelectKBest(chi2, k=int(math.sqrt(X.shape[1]))).fit_transform(X, Y)
print(X.shape)

# ExtraTreesClassifier(bootstrap=False, class_weight='balanced',
#   criterion='gini', max_depth=None, max_features='auto',
#   max_leaf_nodes=None, min_impurity_split=1e-07,
#   min_samples_leaf=1, min_samples_split=2,
#   min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
#   oob_score=False, random_state=None, verbose=0, warm_start=False)

# ExtraTreesClassifier(bootstrap=False,
#           criterion='gini', max_depth=None, max_features=0.75421,class_weight='balanced',
#           max_leaf_nodes=None, min_impurity_decrease=1e-05,
#           min_samples_leaf=2, min_samples_split=2,
#           min_weight_fraction_leaf=0.0, n_estimators=70, n_jobs=-1,
#           oob_score=False, random_state=None, verbose=0, warm_start=False)




model = pkl.load(open("./SerializedModels/ExtraTreesNotDownsampledEnglishStopOnly.pkl", 'rb'))
y_pred = model.predict(X)


printScores(Y, y_pred)

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
import math

numpyRandomSeed = 764
sklearnRandomSeed = 4

file = open('corpus.pkl','rb')
corpus = pkl.load(file)
file.close()
#corpusList = list(corpus.values())
#labels = list(corpus.keys())
corpusList=corpus[corpus.columns[0]].tolist()
labels=corpus[corpus.columns[1]].tolist()



## 22.7% is 1 , 77.27% is 0  this is unbalanced
1-np.sum(labels)/len(labels)



bleed = corpus[corpus[corpus.columns[1]]==1]
bleed = bleed.reset_index()
bleed = bleed.drop('index',axis=1)
print(len(bleed))


no_bleed = corpus[corpus[corpus.columns[1]]==0]
no_bleed = no_bleed.reset_index()
no_bleed = no_bleed.drop('index',axis=1)
print(len(no_bleed))



# # random cut off no_bleed from 765, and make dataset balanced
# np.random.seed(numpyRandomSeed)
# rd_no_bleed = no_bleed.iloc[np.random.choice(len(no_bleed), replace=False, size=len(bleed))]
# all_data =bleed.append(rd_no_bleed, ignore_index=True)
# len(all_data)
#
# corpusList=all_data[all_data.columns[0]].tolist()
# labels=all_data[all_data.columns[1]].tolist()
#
# ## 22.7% is 1 , 77.27% is 0, check if balanced
# 1-np.sum(labels)/len(labels)



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


### before one

cv = TfidfVectorizer(lowercase=True,
                     ngram_range=(1, 3), preprocessor=None, stop_words="english",
                     strip_accents=None, tokenizer=tokenize, vocabulary=None)
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

selector = SelectKBest(chi2, k=int(math.sqrt(X.shape[1])))
X = selector.fit_transform(X, Y)
print(X.shape)
selectedFeatureIndices = selector.get_support()
allFeatures = cv.get_feature_names()
selectedFeatures = np.array(allFeatures)[selectedFeatureIndices]
print(selectedFeatures)
print("Number of vocab terms: %i" % len(selectedFeatures))
pkl.dump(selectedFeatures, open("../FinalModels/SerializedModels/ExtraTreesVocabularyNotDownsampledEnglishStopOnly.pkl", 'wb'), 2)

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




model = 0
results = []
cont = []
kf = KFold(n_splits=10, shuffle=True)
trees = [100]
for t in range(len(trees)):
    fold = 1
    truth = []
    rf_prediction = []
    print("model ", t, ": ExtraTreesClassifier = " + str(trees[t]))
    test_count = 0
    rf = ExtraTreesClassifier(bootstrap=False,
           criterion='gini', max_depth=None, max_features=0.75,
           max_leaf_nodes=None, min_impurity_decrease=1e-051,
           min_samples_leaf=1, min_samples_split=2,
           min_weight_fraction_leaf=0.0, n_estimators=80, n_jobs=-1,
           oob_score=False, random_state=sklearnRandomSeed, verbose=0, warm_start=False)
    for train_idx, test_idx in kf.split(X):
        trainX = X[train_idx]
        trainY = Y[train_idx]
        testX = X[test_idx]
        testY = Y[test_idx]
        truth.append(testY)
        rf.fit(trainX, trainY)
        Y_hat = rf.predict(testX)
        rf_prediction.append(Y_hat)
        print("   Fold %d accuracy: %.2f %%" % (fold, ((np.sum(Y_hat == testY) / len(testY)) * 100.0)))
        fold += 1
    truth = np.concatenate(truth, axis=0)
    rf_prediction = np.concatenate(rf_prediction, axis=0)
    test_results = np.sum(rf_prediction == truth) / len(truth)
    print("     Overall test accuracy: %.2f %%" % (test_results * 100))
    rf = rf.fit(X, Y)
    Y_hat = rf.predict(X)
    train_results = np.sum(Y_hat == Y) / len(Y)
    print("     Overall training accuracy: %.2f %%" % (train_results * 100.0))
    results.append([train_results, test_results])
    cont.append([truth, rf_prediction])

#modelNum = input("Enter Best Model: ")
modelNum = 0
best = cont[int(modelNum)]
rf_ct = pd.crosstab(best[1], best[0], margins=True)
rf_ct.columns = ["Other", "PNA", "Total"]
rf_ct.index = ["Other", "PNA", "Total"]
print()
print("Random Forest")
print(rf_ct)
print()


from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score,f1_score,precision_score,accuracy_score,classification_report
y_true =cont[int(model)][0]
y_pred = cont[int(model)][1]
print(confusion_matrix(y_true, y_pred))
print(f1_score(y_true, y_pred) )
print(recall_score(y_true, y_pred),   )
print(precision_score(y_true, y_pred))
print(accuracy_score(y_true, y_pred))

print(classification_report(y_true, y_pred))

model = ExtraTreesClassifier(bootstrap=False,
           criterion='gini', max_depth=None, max_features=0.75,
           max_leaf_nodes=None, min_impurity_decrease=1e-051,
           min_samples_leaf=1, min_samples_split=2,
           min_weight_fraction_leaf=0.0, n_estimators=80, n_jobs=-1,
           oob_score=False, random_state=sklearnRandomSeed, verbose=0, warm_start=False)
model = model.fit(X, Y)

pkl.dump(model, open("../FinalModels/ExtraTreesNotDownsampledEnglishStopOnly.pkl", 'wb'), 2)
y_pred = model.predict(X)

print(f1_score(Y, y_pred))
print(recall_score(Y, y_pred))
print(precision_score(Y, y_pred))
print(accuracy_score(Y, y_pred))


import nltk
from nltk import word_tokenize

from sklearn.model_selection import KFold

import glob

from Features import *
from Classify import *

class ReadFile(object):
    def __init__(self):
        self.corpus = []
        self.corpus_neg = []
        self.corpus_pos = []

    def readFile(self):
        path = "C:/Users/Robee Khyra Te/Documents/GitHub/machlrn/Final Project/tokens/*"

        for folder in glob.glob(path):
            if "neg" in folder:
                for textfile in glob.glob(folder.replace("\\neg", "/neg/*")):
                    with open(textfile.replace("\\", "/")) as f:
                        text = f.read()
                    self.corpus_neg.append((text, 0))
            elif "pos" in folder:
                for textfile in glob.glob(folder.replace("\\pos", "/pos/*")):
                    with open(textfile.replace("\\", "/")) as f:
                        text = f.read()
                    self.corpus_pos.append((text, 1))

        print(len(self.corpus_neg))
        print(len(self.corpus_pos))

    def getCorpusNeg(self):
        return self.corpus_neg

    def getCorpusPos(self):
        return self.corpus_pos

rf = ReadFile()
rf.readFile()

f = Features()
negativeDocuments = f.getListWithNegation(rf.getCorpusNeg())
positiveDocuments = f.getListWithNegation(rf.getCorpusPos())

kf = KFold(3)

accuracyNB = 0
accuracySVM = 0
indexOfTest = 0

for train, test in kf.split(rf.getCorpusNeg()): #Get training indices and testing indices
    negDoc = []
    posDoc = []

    for index in train:
        negDoc.append(negativeDocuments[index])
        posDoc.append(positiveDocuments[index])

    negativeUnigrams = f.getUnigram(negDoc)
    positiveUnigrams = f.getUnigram(posDoc)

    negativeFeatures = f.getChosenFeatures(negDoc, type = 'presence')
    positiveFeatures = f.getChosenFeatures(posDoc, type = 'presence')

    f.features['unigram'] = []

    c = Classify()

    x_train, x_test, y_train, y_test = c.prepare(negativeFeatures, positiveFeatures, indexOfTest)

    accuracyNB += c.naiveBayesClassifier(x_train, x_test, y_train, y_test)

    accuracySVM += c.SVMClassifier(x_train, x_test, y_train, y_test)

    indexOfTest += 1

print(accuracyNB / 3 * 100)
print(accuracySVM / 3 * 100)

#f.getBigram(rf.getCorpus())
#f.getUnigramPos()
#f.getUnigramAdjective()
#f.getUnigramPosition()\
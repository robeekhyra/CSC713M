from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

import numpy as np

class Classify(object):
    def __init__(self):
        self.number_fold = 3
        self.splitPerFold = 700 // 3

        self.batch = {}
        
        self.batch['X'] = []
        self.batch['Y'] = []

    def prepare(self, corpus):
        #Create batches
        for i in range(self.number_fold):
            neg = corpus[i * self.splitPerFold : i * self.splitPerFold + self.splitPerFold]
            print(neg)
            pos = corpus[i * self.splitPerFold + 699 : i * self.splitPerFold + self.splitPerFold + 699]
            print(pos)
            self.batch['X'].append(np.concatenate(neg, pos))
            self.batch['Y'].append(np.zeros(self.splitPerFold), np.ones(self.splitPerFold))
            print(self.batch['X'][0])
            print(self.batch['Y'])

    def naiveBayesClassifier(self):
        accuracy = 0

        #ITO NA, RUN 3 TIMES, GET AVERAGE
        for n in range(number_fold):
            x_train = []
            y_train = []
            x_test = []
            y_test = []

            MultinomialNB().fit(x_train, y_train)
            accuracy += accuracy_score(MultinomialNB().predict(x_test), y_test)
            accuracy_naivebayes = accuracy / number_fold

        print(accuracy_naivebayes)

    def SVM(self):
        return null

    def maxEntropy(self):
        return null
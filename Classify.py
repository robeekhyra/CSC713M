from sklearn.model_selection import KFold
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

import numpy as np

class Classify(object):
    def __init__(self):
        self.number_fold = 3

    def splitToMiniBatches(self, negative, positive, indexOfTest):
        batch = {'fold' : [], 'X' : [], 'Y' : []}

        split = len(negative) // self.number_fold

        #Create mini batches
        for i in range(self.number_fold):
            mergedlist = []

            batch['fold'].append(i)
            
            #Split to equal sized batch, maintaining balance from neg and pos
            neg = negative[i * split : i * split + split]
            #print(len(neg))
            pos = positive[i * split : i * split + split]
            #print(len(pos))
            
            mergedlist.extend(neg)
            mergedlist.extend(pos)
            batch['X'].append(mergedlist)
            #print(len(self.batch['X'][i]))

            batch['Y'].append(np.append(np.zeros(split), np.ones(split)))
            #print(len(self.batch['Y'][i]))

        batch['fold'] = np.array(batch['fold'])
        batch['X'] = np.array(batch['X'])
        batch['Y'] = np.array(batch['Y'])

        x_train = batch['X'][batch['fold'] != i]
        y_train = batch['Y'][batch['fold'] != i]

        x_test = batch['X'][i]
        y_test = batch['Y'][i]

        #print(x_test.shape)
        #print(y_test.shape)

        return x_train, x_test, y_train, y_test
    
    def naiveBayesClassifier(self, x_train, x_test, y_train, y_test):
        bernoulliNB = BernoulliNB()
        bernoulliNB.fit(x_train.reshape(x_train.shape[0] * x_train.shape[1], -1), np.ravel(y_train.reshape(y_train.shape[0] * y_train.shape[1], -1)))
        accuracy = accuracy_score(bernoulliNB.predict(x_test), y_test)
        
        #print("Accuracy for 1 fold in Naive Bayes: ", accuracy)

        return accuracy

    def SVMClassifier(self, x_train, x_test, y_train, y_test):
        linearSVM = LinearSVC()
        linearSVM.fit(x_train.reshape(x_train.shape[0] * x_train.shape[1], -1), np.ravel(y_train.reshape(y_train.shape[0] * y_train.shape[1], -1)))
        accuracy = accuracy_score(linearSVM.predict(x_test), y_test)
    
        #print("Accuracy for 1 fold in SVM: ", accuracy)

        return accuracy
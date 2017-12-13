import numpy as np
import nltk
from nltk.util import ngrams
from collections import Counter

class Features(object):
    def __init__(self):
        self.negate = "NOT_"

        self.negation_list = ["no", "not", "n\'t"]
        self.punctuation_list = [".", ",", "?", "!", ";"]

        self.features = {}

        self.features['unigram'] = []
        self.features['bigram'] = []
        self.features['unigram_bigram'] = []
        self.features['topunigrams'] = []
        self.features['unigram_pos'] = []
        self.features['adjectives'] = []
        self.features['unigram_position'] = []

    def getListWithNegation(self, corpus):
        all_document = []

        for document, category in corpus:
            all_words = []
            tokens = nltk.word_tokenize(document)
            i = 0
            while i < len(tokens):
                if tokens[i] in self.negation_list:
                    all_words.append(tokens[i])
                    i += 1
                    while i < len(tokens):
                        if tokens[i] not in self.punctuation_list:
                            a = tokens[i] + self.negate
                            all_words.append(a)
                        else:
                            break
                        i += 1
                else:
                    all_words.append(tokens[i])
                i += 1

            all_document.append((all_words, category))

        #print(all_document)

        #print(all_words)

        return all_document

    def getUnigram(self, all_document):
        unigrams = {}

        for words_in_document, category in all_document:
            for word in words_in_document:
                unigrams[word] = unigrams.get(word, 0) + 1

        for word, count in unigrams.items():
            if count >= 4: #SABI SA DOCU AT LEAST 4
                self.features['unigram'].append(word)

        #print(len(self.features['unigram']))

        return self.features['unigram']

    def getChosenFeatures(self, all_document, type = 'None'):
        chosen_features = np.zeros((len(all_document), len(self.features['unigram'])))

        i = 0
        for words_in_document, category in all_document:
            if type == 'frequency':
                frequencies = Counter(words_in_document)
                for word, count in frequencies.items():
                    try:
                        chosen_features[i][self.features['unigram'].index(word)] = count
                    except: #PAG WALA
                        pass
            elif type == 'presence':
                for word in set(words_in_document):
                    try:
                        chosen_features[i][self.features['unigram'].index(word)] = 1
                    except: #PAG WALA
                        pass
            i += 1

        #print(chosen_features)

        return chosen_features

    def getBigram(self, all_document):
        bigrams = {}
        words = []

        for words_in_document, category in all_document:
            #tokens = nltk.word_tokenize(words_in_document)
            words += ngrams(words_in_document, 2)
        
        for word in words:
            bigrams[word] = bigrams.get(word, 0) + 1

        for word, count in bigrams.items():
            if count >= 7: #SABI SA DOCU AT LEAST 7
                self.features['bigram'].append(word)

        #print(self.features['bigram'][:100])
        return self.features['bigram'][:16165]

    def getChosenFeaturesBigram(self, all_document):
        chosen_features = np.zeros((len(all_document), len(self.features['bigram'])))

        i = 0
        for words_in_document, category in all_document:
            words = []
            #tokens = nltk.word_tokenize(words_in_document)
            words += ngrams(words_in_document, 2)
            for word in set(words):
                try:
                    chosen_features[i][self.features['bigram'].index(word)] = 1
                except: #PAG WALA
                    pass
            i += 1

        #print(chosen_features)
        return chosen_features

    def getUnigramBigram(self):
        self.features[unigram_bigram] = self.features['unigram'] + self.features['bigram']

    def getChosenFeaturesUniBigram(self, all_document, features):
        chosen_features = np.zeros((len(all_document), len(features)))

        i = 0
        for words_in_document, category in all_document:
            words = []
            #tokens = nltk.word_tokenize(words_in_document)
            words += ngrams(words_in_document, 2)
            for word in set(words):
                try:
                    chosen_features[i][features.index(word)] = 1
                except: #PAG WALA
                    pass
            i += 1

        #print(chosen_features)
        return chosen_features

    def getTopUnigrams(self, all_document):
        frequencies = Counter(self.features['unigram'])
        self.features['topunigrams'] = sorted(frequencies, key=frequencies.get, reverse=True)[:2633]

        chosen_features = np.zeros((len(all_document), len(self.features['topunigrams'])))

        i = 0
        for words_in_document, category in all_document:
            for word in set(words_in_document):
                try:
                    chosen_features[i][self.features['topunigrams'].index(word)] = 1
                except: #PAG WALA
                    pass
            i += 1

        #print(chosen_features)
        return chosen_features

    def getUnigramPos(self, all_document):
        unigramspos = {}

        for words_in_document, category in all_document:
            for word, pos in nltk.pos_tag(words_in_document):
                unigramspos[word + "-" + pos] = unigramspos.get(word + "-" + pos, 0) + 1

        for word, count in unigramspos.items():
            if count >= 4:
                self.features['unigram_pos'].append(word)

        #print(len(self.features['unigram_pos']))
        
        return self.features['unigram_pos']

    def getChosenFeaturesPos(self, all_document):
        chosen_features = np.zeros((len(all_document), len(self.features['unigram_pos'])))

        i = 0
        for words_in_document, category in all_document:
            for word, pos in nltk.pos_tag(words_in_document):
                try:
                    chosen_features[i][self.features['unigram_pos'].index(word + "-" + pos)] = 1
                except: #PAG WALA
                    pass
            i += 1

        #print(chosen_features)
        return chosen_features

    def getUnigramAdjective(self, all_document):
        for words_in_document, category in all_document:
            for word, pos in nltk.pos_tag(words_in_document):
                if pos in ['JJ', 'JJR', 'JJS']:
                    self.features['adjectives'].append(word)
        
        return self.features['adjectives']
    
    def getChosenFeaturesAdjective(self, all_documents):
        chosen_features = np.zeros((len(all_document), len(self.features['adjectives'])))
        
        i = 0
        for words_in_document, category in all_document:
            for word in set(words_in_document):
                try:
                    chosen_features[i][self.features['adjectives'].index(word)] = 1
                except: #PAG WALA
                    pass
            i += 1

        #print(chosen_features)
        return chosen_features

    def getUnigramPosition(self, all_document):
        unigramsposition = {}
        
        for words_in_document, category in all_document:
            splitPerQtr = len(words_in_document) // 4
            i = 0
            position = 0
            for word in words_in_document: #START
                if i < splitPerQtr:
                    unigramsposition[word + "-" + str(position)] = unigramsposition.get(word + "-" + str(position), 0) + 1
                if i >= splitPerQtr and i < len(words_in_document) - splitPerQtr: #MIDDLE
                    position = 1
                elif i <= len(words_in_document) - splitPerQtr: #END
                    position = 2
                i += 1 

        for word, count in unigramsposition.items():
            if count >= 4:
                self.features['unigram_position'].append(word)

        #print(len(self.features['unigram_position']))
        
        return self.features['unigram_position']

    def getChosenFeaturesPosition(self, all_document):
        chosen_features = np.zeros((len(all_document), len(self.features['unigram_position'])))

        i = 0
        for words_in_document, category in all_document:
            splitPerQtr = len(words_in_document) // 4
            j = 0
            position = 0
            for word in set(words_in_document):
                try:
                    if j < splitPerQtr:
                        chosen_features[i][self.features['unigram_position'].index(word + "-" + str(position))] = 1
                    if j >= splitPerQtr and j < len(words_in_document) - splitPerQtr: #MIDDLE
                        position = 1
                    elif j <= len(words_in_document) - splitPerQtr: #END
                        position = 2 
                    j += 1
                except: #PAG WALA
                    j += 1
                    pass
            i += 1

        #print(chosen_features)
        
        return chosen_features
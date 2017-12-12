import nltk
from nltk.util import ngrams
from collections import Counter

class Features(object):
    def __init__(self):
        self.negate = "NOT_"

        #self.negation_list = ["no", "not", "ain\'t", "isn\'t", "aren\'t", "wasn\'t", "weren\'t", "can\'t", "couldn\'t", "won\'t", "wouldn\'t", "shouldn\'t", "doesn\'t", "don\'t", "didn\'t", "hasn\'t", "haven\'t", "hadn\'t"]
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

        self.all_words = []

    def getListWithNegation(self, corpus):
        for document in corpus:
            tokens = nltk.word_tokenize(document)
            i = 0
            while i < len(tokens):
                if tokens[i] in self.negation_list:
                    i += 1
                    while i < len(tokens):
                        if tokens[i] not in self.punctuation_list:
                            a = tokens[i] + self.negate
                            self.all_words.append(a)
                        else:
                            break
                        i += 1
                else:
                    self.all_words.append(tokens[i])
                i += 1

        #print(self.all_words)

        return self.all_words

    def getUnigram(self, all_words):
        unigrams = {}
        
        for word in all_words:
            unigrams[word] = unigrams.get(word, 0) + 1

        for word, count in unigrams.items():
            if count >= 4:
                self.features['unigram'].append(word)

        print(len(self.features['unigram']))

    def getBigram(self, corpus):
        bigrams = {}
        words = []

        for document in corpus:
            tokens = nltk.word_tokenize(document)
            words += ngrams(tokens, 2)
        
        for word in words:
            bigrams[word] = bigrams.get(word, 0) + 1

        for word, count in bigrams.items():
            if count >= 7:
                self.features['bigram'].append(word)

        self.features['bigram'] = self.features['bigram'][:16165]

        #print(self.features['bigram'])

    def getUnigramBigram(self):
        self.features[unigram_bigram] = self.features['unigram'] + self.features['bigram']

    def getTopUnigrams(self):
        frequencies = Counter(self.features['unigram'])
        self.features['topunigrams'] = sorted(frequencies, key=frequencies.get, reverse=True)[:2633]

    def getUnigramPos(self):
        unigrams = {}

        for word, pos in nltk.pos_tag(self.features['unigram']):
            unigrams[word + "-" + pos] = unigrams.get(word + "-" + pos, 0) + 1

        for word, count in unigrams.items():
            if count >= 4:
                self.features['unigram_pos'].append(word)

        print(len(self.features['unigram_pos']))

    def getUnigramAdjective(self):
        for word, pos in nltk.pos_tag(self.features['unigram_pos']):
            if pos in ['JJ', 'JJR', 'JJS']:
                self.features['adjectives'].append(word)

    def getUnigramPosition(self):
        return null
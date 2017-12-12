import nltk
from nltk import word_tokenize

import glob
import random

from Features import *
from Classify import *

class ReadFile(object):
    def __init__(self):
        self.corpus = []

    def readFile(self):
        path = "C:/Users/Robee Khyra Te/Documents/GitHub/machlrn/Final Project/tokens/*"

        for folder in glob.glob(path):
            for textfile in glob.glob(folder.replace("\\", "/") + "/*"):
                with open(textfile.replace("\\", "/")) as f:
                    text = f.read()
                self.corpus.append(text)

    def getCorpus(self):
        return self.corpus

rf = ReadFile()
rf.readFile()
f = Features()
f.getUnigram(f.getListWithNegation(rf.getCorpus()))
f.getBigram(rf.getCorpus())
c = Classify()
c.prepare(rf.getCorpus())
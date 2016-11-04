from sys import argv
import wikipedia
from unidecode import unidecode
import re
import pickle
import glob
import os

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag


class article:

    def __init__(self, articleName):
        # Kind of downloading article
        wikiPage = wikipedia.page(articleName)
        # Cleaning, finding topics and plain text from the article
        self.extract(wikiPage)

    def extract(self, wikiPage):
        # self.text and self.topics
        self.extractText(wikiPage)
        # self.sentences
        self.extractSentences()
        # self.words, self.sentenceID
        self.extractWords()

    def extractText(self, wikiPage):
        self.topics = {}
        self.text = ""
        # prelim cleaning, translitting non unicode stuff
        temp = unidecode(wikiPage.content)
        # remove all paranthesized stuff
        re.sub(r'\([^\)]*\)', '', temp)
        re.sub(r'\[[^\]]*\]', '', temp)
        topicsSearch = re.findall(r'=+[^=]*=+', temp)
        self.sections = []
        for t in re.findall(r'=+[^=]*=+', temp):
            match = re.search(r'=+([^=]*)=+', t)
            if match != None and match.group(1) != '':
                self.sections.append(match.group(1).strip())
        self.text += unidecode(wikiPage.summary).lower()
        for s in self.sections:
            if wikiPage.section(s) != None:
                self.topics[s] = unidecode(wikiPage.section(s)).lower()
                self.text += unidecode(wikiPage.section(s)).lower()

    def extractSentences(self):
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        self.sentences = tokenizer.tokenize(self.text)

    def extractWords(self):
        sentences = self.sentences
        sentenceID = []
        stop = set(stopwords.words('english'))
        wnl = WordNetLemmatizer()
        words = []
        for i in range(len(sentences)):
            s = sentences[i]
            # replace all non alphabetic stuff with => "u.s.a" to "usa" and "12.69" to ""
            rawWords = re.sub(r"[^a-zA-Z\s]+", "", s).split()
            for w in rawWords:
                # for each word in current sentence
                if w not in stop:
                    # if it is not a stop word, we need its sentenceID
                    sentenceID.append(i)
                    pos = pos_tag(w)[0][1][0].lower()
                    if pos in ['a', 'n', 'v']:
                        words.append(wnl.lemmatize(w, pos))
                    else:
                        words.append(wnl.lemmatize(w))
        self.words = words
        self.sentenceID = sentenceID

    def getText(self):
        return self.text

    def getTopics(self):
        return self.topics

    def getSentences(self):
        return self.sentences

    def getWords(self):
        return self.words

    def getSentenceIDs(self):
        return self.sentenceID


def cachedArticle(articleName):
    with open('./Corpus/' + articleName + '.pkl', 'rb') as f:
        return pickle.load(f)


def allCachedArticles():
    allArticles = []
    os.chdir("./Corpus")
    for f in glob.glob("*.pkl"):
        allArticles.append(pickle.load(f))
    return allArticles
if __name__ == "__main__":
    someArticle = cachedArticle(argv[1])
    print(someArticle.getText())
    print(someArticle.getTopics().keys())
    print(len(someArticle.getTopics().keys()))

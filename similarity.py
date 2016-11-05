import preprocess
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
import re
from unidecode import unidecode
from sys import argv
import matplotlib.pyplot as plt
from pprint import pprint
from collections import Counter
import numpy as np


def readWordVectors():
    # Hard Coded File Name
    wordVectors = {}
    with open('wordVecs.txt', 'rb') as f:
        lines = f.readlines()
        i = 0
        for l in lines:
            l = l.decode('utf-8')
            t = unidecode(l).split()
            try:
                wordVectors[t[0]] = [float(j) for j in t[1:]]
            except:
                pass
    return wordVectors


def addVectors(vectors):
    vectorLength = len(vectors[0])
    s = [0 for i in range(vectorLength)]
    for v in vectors:
        for i in range(vectorLength):
            s[i] += v[i]
    return s


def dotProduct(v1, v2):
    return sum([x1 * x2 for x1, x2 in zip(v1, v2)])


def cosineProduct(v1, v2):
    return dotProduct(v1, v2) / (dotProduct(v1, v1) * dotProduct(v2, v2))**0.5


def computeConcepts(window):
    global wordVectors
    vec = []
    for l in window:
        try:
            vec.append(wordVectors[l])
        except:
            pass
    return addVectors(vec)


def computeSimilarity(article, windowWidth=20):
    global wordVectors
    words = article.getWords()
    sentenceID = article.getSentenceIDs()
    sentenceCount = len(article.getSentences())

    lexicalScores = [0 for i in range(sentenceCount)]

    for i in range(windowWidth, len(words) - windowWidth + 1):
        leftWindow = words[i - windowWidth:i]
        rightWindow = words[i:i + windowWidth]
        leftConcept = computeConcepts(leftWindow)
        rightConcept = computeConcepts(rightWindow)
        lexicalScores[sentenceID[i]] += cosineProduct(leftConcept, rightConcept)

    lexicalScores = [lexicalScores[i] / sentenceID.count(i) for i in range(sentenceCount)]
    return lexicalScores


def computeSimilarity1(article, windowWidth=30):
    global wordVectors
    words = article.getWords()
    sentenceID = article.getSentenceIDs()
    sentenceCount = len(article.getSentences())

    lexicalScores = [0 for i in range(sentenceCount)]

    for i in range(windowWidth, len(words) - windowWidth + 1):
        visitedConcepts = words[:i - windowWidth]
        leftWindow = words[i - windowWidth:i]
        rightWindow = words[i:i + windowWidth]
        newLeftCount = 0
        newRightCount = 0
        for l in leftWindow:
            if l not in visitedConcepts:
                visitedConcepts.append(l)
                newLeftCount += 1
        for r in rightWindow:
            if r not in visitedConcepts:
                visitedConcepts.append(r)
                newRightCount += 1
        lexicalScores[sentenceID[i]] += (newLeftCount + newRightCount) / (2 * windowWidth)

    lexicalScores = [lexicalScores[i] / sentenceID.count(i) for i in range(sentenceCount)]
    return lexicalScores


def computeSimilarity2(article, w=20, k=10):
    global wordVectors
    words = article.getWords()
    sentenceID = article.getSentenceIDs()
    sentenceCount = len(article.getSentences())

    lexicalScores = [0 for i in range(len(words))]
    windowWidth = k * w
    for i in range(windowWidth, len(words) - windowWidth + 1, w):
        leftWindow = words[i - windowWidth:i]
        rightWindow = words[i:i + windowWidth]
        leftConcept = computeConcepts(leftWindow)
        rightConcept = computeConcepts(rightWindow)
        lexicalScores[i] += cosineProduct(leftConcept, rightConcept)

    return lexicalScores


def computeSimilarity3(article, w=20, k=10):
    # Jaccard similarity
    words = article.getWords()
    sentenceID = article.getSentenceIDs()
    sentenceCount = len(article.getSentences())
    lexicalScores = []
    windowWidth = k * w
    for i in range(windowWidth, len(words) - windowWidth + 1, w):
        leftWindow = words[i - windowWidth:i]
        rightWindow = words[i:i + windowWidth]
        leftConcept = set(leftWindow)
        rightConcept = set(rightWindow)
        allWindowWords = set(leftWindow + rightWindow)
        lexicalScores.append(len(leftConcept.intersection(rightConcept)) / len(allWindowWords))

    return lexicalScores


def computeSimilarity4(article, w=20, k=10):
    global wordVectors
    words = article.getWords()
    sentenceID = article.getSentenceIDs()
    sentenceCount = len(article.getSentences())
    lexicalScores = []
    windowWidth = k * w
    for i in range(windowWidth, len(words) - windowWidth + 1, w):
        leftWindow = words[i - windowWidth:i]
        rightWindow = words[i:i + windowWidth]
        leftConcept = Counter(leftWindow)
        rightConcept = Counter(rightWindow)
        allWindowWords = list(set(leftWindow + rightWindow))
        sim = 0
        lSquare = 0
        rSquare = 0
        for w in allWindowWords:
            sim += leftConcept[w] * rightConcept[w]
            lSquare += leftConcept[w]**2
            rSquare += rightConcept[w]**2
        lexicalScores.append(sim / (lSquare * rSquare)**0.5)
    return lexicalScores

def computeSimilarity5(article, w=20, k=10):
    global wordVectors
    words = article.getWords()
    sentenceID = article.getSentenceIDs()
    sentenceCount = len(article.getSentences())

    lexicalScores = []
    windowWidth = k * w
    for i in range(windowWidth, len(words) - windowWidth + 1, w):
        leftWindow = words[i - windowWidth:i]
        rightWindow = words[i:i + windowWidth]
        left_tensor = [wordVectors[word] for word in leftWindow]
        right_tensor = [wordVectors[word] for word in rightWindow]
        lexicalScores.append(np.tensordot(left_tensor, right_tensor))
    return lexicalScores

def constructGraph(lexicalScores):
    plt.plot(lexicalScores)
    plt.show()


if __name__ == "__main__":
    wordVectors = readWordVectors()
    article = preprocess.cachedArticle(argv[1])
    lexicalScores = computeSimilarity5(article, int(argv[2], int(argv[3])))
    constructGraph(lexicalScores)

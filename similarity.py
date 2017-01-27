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


def getWordVector(word):
    global wordVectors
    try:
        return wordVectors[word]
    except:
        return [0 for i in range(50)]


def dotProduct(v1, v2):
    return sum([x1 * x2 for x1, x2 in zip(v1, v2)])


def cosineProduct(v1, v2):
    if dotProduct(v2, v2) == 0 or dotProduct(v1, v1) == 0:
        return 0
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


def computeTensor(window):
    global wordVectors
    temp = [wordVectors[l] for l in window if l in wordVectors.keys()]
    mean = [i / len(temp) for i in addVectors(temp)]
    tensor = []
    for l in window:
        try:
            tensor.append(wordVectors[l])
        except:
            tensor.append(mean)
    return tensor

# Using word vectors


def computeSimilarity(article, w, k):
    print("Obselete Similarity")
    exit(0)
    global wordVectors
    words = article.getWords()
    sentenceID = article.getSentenceIDs()
    sentenceCount = len(article.getSentences())

    lexicalScores = [0 for i in range(sentenceCount)]
    windowWidth = w * k
    for i in range(windowWidth, len(words) - windowWidth + 1):
        leftWindow = words[i - windowWidth:i]
        rightWindow = words[i:i + windowWidth]
        leftConcept = computeConcepts(leftWindow)
        rightConcept = computeConcepts(rightWindow)
        lexicalScores[sentenceID[i]] += cosineProduct(leftConcept, rightConcept)

    lexicalScores = [lexicalScores[i] / sentenceID.count(i) for i in range(sentenceCount)]
    return lexicalScores

# Using new concepts


def computeSimilarity1(article, w, k):
    print("Obselete Similarity")
    exit(0)
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

# Using wordVectors


def computeSimilarity2(article, w, k):
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

# Using Jaccard similarity


def computeSimilarity3(article, w, k):
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

# Using commonWords between windows


def computeSimilarity4(article, w, k):
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


def computeSimilarity5(article, w, k):
    global wordVectors
    wordVectors = readWordVectors()
    words = article.getWords()
    sentenceID = article.getSentenceIDs()
    sentenceCount = len(article.getSentences())

    lexicalScores = []
    windowWidth = k * w
    for i in range(windowWidth, len(words) - windowWidth + 1, w):
        leftWindow = words[i - windowWidth:i]
        rightWindow = words[i:i + windowWidth]
        left_tensor = computeTensor(leftWindow)
        right_tensor = computeTensor(rightWindow)
        print(len(left_tensor), len(left_tensor[0]))
        print(len(right_tensor), len(right_tensor[0]))
        lexicalScores.append(np.tensordot(left_tensor, right_tensor) / (np.tensordot(left_tensor, left_tensor) * np.tensordot(right_tensor, right_tensor))**0.5)
    return lexicalScores


def computeSimilarity6(article, w, k):
    global wordVectors
    wordVectors = readWordVectors()
    words = article.getWords()
    sentenceID = article.getSentenceIDs()
    sentenceCount = len(article.getSentences())

    lexicalScores = []
    windowWidth = k * w
    for i in range(windowWidth, len(words) - windowWidth + 1, w):
        leftWindow = words[i - windowWidth:i]
        rightWindow = words[i:i + windowWidth]
        s = 0
        for l in leftWindow:
            s += max([cosineProduct(getWordVector(l), getWordVector(r)) for r in rightWindow])
        lexicalScores.append(s)
    return lexicalScores

if __name__ == "__main__":
    # wordVectors = readWordVectors()
    article = preprocess.cachedArticle(argv[1])
    topicIndices = []
    temp = 0
    for s in article.getTopicWiseSentences():
        topicIndices.append(len(s) + temp)
        temp += len(s)
    w = 20
    k = 10
    try:
        w = int(argv[2])
        k = int(argv[3])
    except:
        pass
    windowWidth = w * k
    lexicalScores = computeSimilarity5(article, w, k)
    f = plt.figure(2)
    plt.plot(lexicalScores)
    plt.show()
    # words = article.getWords()
    # sentenceID = article.getSentenceIDs()
    # # plt.plot([i for i in range(windowWidth, len(words) - windowWidth + 1, w)],lexicalScores)
    # # for i in range(windowWidth, len(words) - windowWidth + 1, w):
    # #     plt.plot((sentenceID[i], sentenceID[i]), (0, 0.2), 'r')
    # for i in topicIndices:
    #     plt.plot((i, i), (0, 0.2), 'r')
    # # print(len(lexicalScores))
    # # print(len(range(windowWidth, len(words) - windowWidth + 1, w)))
    # plt.plot([sentenceID[i] for i in range(windowWidth, len(words) - windowWidth + 1, w)],
    #          [lexicalScores[i] for i in range(len(lexicalScores))])
    # plt.show()

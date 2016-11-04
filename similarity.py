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


def extractSentences(text):
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    return tokenizer.tokenize(text)


def extractWords(text):
    sentences = extractSentences(text)
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
    return words, sentenceID


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


def computeSimilarity(text, windowWidth=20):
    global wordVectors
    words, sentenceID = extractWords(text)
    wordVectors = readWordVectors()
    sentenceCount = max(sentenceID) + 1

    lexicalScores = [0 for i in range(sentenceCount)]

    for i in range(windowWidth, len(words) - windowWidth + 1):
        leftWindow = words[i - windowWidth:i]
        rightWindow = words[i:i + windowWidth]
        leftConcept = computeConcepts(leftWindow)
        rightConcept = computeConcepts(rightWindow)
        lexicalScores[sentenceID[i]] += cosineProduct(leftConcept, rightConcept)

    lexicalScores = [lexicalScores[i] / sentenceID.count(i) for i in range(sentenceCount)]
    return lexicalScores


def computeSimilarity1(text, windowWidth=30):
    global wordVectors
    words, sentenceID = extractWords(text)
    wordVectors = readWordVectors()
    sentenceCount = max(sentenceID) + 1

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


def computeSimilarity2(text, w=20, k=10):
    global wordVectors
    words, sentenceID = extractWords(text)
    wordVectors = readWordVectors()
    lexicalScores = [0 for i in range(len(words))]
    windowWidth = k * w
    for i in range(windowWidth, len(words) - windowWidth + 1, w):
        leftWindow = words[i - windowWidth:i]
        rightWindow = words[i:i + windowWidth]
        leftConcept = computeConcepts(leftWindow)
        rightConcept = computeConcepts(rightWindow)
        lexicalScores[i] += cosineProduct(leftConcept, rightConcept)

    return lexicalScores


def computeSimilarity3(text, w=20, k=10):
    #Jaccard similarity
    global wordVectors
    words, sentenceID = extractWords(text)
    # wordVectors = readWordVectors()
    lexicalScores = []
    windowWidth = k * w
    for i in range(windowWidth, len(words) - windowWidth + 1, w):
        leftWindow = words[i - windowWidth:i]
        rightWindow = words[i:i + windowWidth]
        leftConcept = set(leftWindow)
        rightConcept = set(rightWindow)
        allWindowWords = set(leftWindow + rightWindow)
        lexicalScores.append(len(leftConcept.intersection(rightConcept))/len(allWindowWords))

    return lexicalScores

def computeSimilarity4(text, w=20, k=10):
    global wordVectors
    words, sentenceID = extractWords(text)
    # wordVectors = readWordVectors()
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
        lexicalScores.append(sim/(lSquare*rSquare)**0.5)

    return lexicalScores


# def computeSimilarity3(text, windowWidth=20):
#     sentences = extractSentences(text)
#     words, sentenceID = extractWords(text)
#     unique_words = list(set(words))
#     wordLists = [[] for s in sentences]
#     for i in range(len(words)):
#         wordLists[sentenceID[i]].append(words[i])
#     lexicalScores = []
#     for gap in range(1, len(wordLists)):
#         current_blocksize = min(gap, windowWidth, len(wordLists) - gap)
#         leftWindow = [word]
#         rightWindow =
#         block1_vector = Counter()
#         block2_vector = Counter()

#         for j in range(1, current_blocksize):
#             block1_vector += Counter(wordLists[gap + j - current_blocksize])
#             block2_vector += Counter(wordLists[gap + j])
#         val = 0.0
#         print(block1_vector, block2_vector)
#         block1_square = 0
#         block2_square = 0
#         for word_token in unique_words:
#             val += (block1_vector[word_token] * block2_vector[word_token])
#             print(word_token)
#             block1_square += (block1_vector[word_token]**2)
#             block2_square += (block2_vector[word_token]**2)
#             print(block2_square, block1_square)
#         lexicalScores.append(val / (block1_square * block2_square)**0.5)
#     return lexicalScores


def constructGraph(lexicalScores):
    plt.plot(lexicalScores)
    plt.show()
if __name__ == "__main__":
    article = preprocess.article(argv[1])
    lexicalScores = computeSimilarity3(article.getText(), int(argv[2], int(argv[3])))
    constructGraph(lexicalScores)

from sys import argv
import preprocess
import similarity
import textTiling
import numpy as np
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag


def cleanText(s):
    # TODO: Also add all single letters
    stop = set(stopwords.words('english'))
    wnl = WordNetLemmatizer()
    words = []
    # replace all non alphabetic stuff with => "u.s.a" to "u s a" and "12.69" to ""
    rawWords = re.sub(r"[^a-zA-Z\s]+", " ", s).split()
    for w in rawWords:
        # for each word in current sentence
        if w not in stop:
            pos = pos_tag(w)[0][1][0].lower()
            if pos in ['a', 'n', 'v']:
                words.append(wnl.lemmatize(w, pos))
            else:
                words.append(wnl.lemmatize(w))
    return words


def similarityMeasure(s1, s2):
    words1 = set(s1)
    words2 = set(s2)
    allWords = words1.union(words2)
    measure = len(words1.intersection(words2)) / len(allWords)
    return measure

# Takes the text and returns matrix of similarity measure between sentences


def buildSimilarityMatrix(sentences):
    n = len(sentences)
    similarity = [[0 for x in range(n)] for y in range(n)]
    for i in range(n):
        for j in range(n):
            similarity[i][j] = similarityMeasure(sentences[i], sentences[j])

    for i in range(n):
        rowSum = sum(similarity[:][i])
        for j in range(n):
            similarity[i][j] = similarity[i][j] / rowSum
            if i == j:
                similarity[i][j] = 0
    similarity = np.array(similarity)

    return similarity


def pageRank(similarity, damping, error):
    n = len(similarity[0])
    initialRanks = np.array([0 for x in range(n)])
    finalRanks = np.array([0 for x in range(n)])

    for i in range(n):
        initialRanks[i] = 1 / n
    n_iterations = 0
    delta = 1.0
    while delta > error:
        finalRanks = (damping * similarity.dot(initialRanks)) + (1 - damping) / n
        # print(finalRanks)
        delta = sum(abs(finalRanks - initialRanks))
        # print(delta)
        n_iterations += 1
        initialRanks, finalRanks = finalRanks, initialRanks

    return finalRanks, n_iterations


def textRankSummary(sentences, fraction):
    cleanedSentences = [cleanText(i) for i in sentences]
    similarityMatrix = buildSimilarityMatrix(cleanedSentences)

    finalRanks, iters = pageRank(similarityMatrix, damping=0.85, error=0.001)
    topIndices = np.argsort(finalRanks)[-1 * int(fraction * len(sentences)):]
    topIndices = sorted(topIndices)
    topSentences = [sentences[i] for i in topIndices]
    # also return topCleanSentences = [cleanedSentences[i] for i in topIndices] why the  fuck would you want to return this cleaned shit??
    return topSentences

# returns list of list of sentences


def getTiles(article, boundaryPoints, w, k):
    tiles = []
    start = 0
    sentences = article.getSentences()
    windowWidth = w * k
    sentenceID = article.getSentenceIDs()
    for b in boundaryPoints:
        tiles.append(sentences[start: sentenceID[windowWidth + w * b]])
    tiles.append(sentences[start:])
    return tiles


def summarize(article, fraction):
    lexicalScores = similarity.computeSimilarity3(article, w=20, k=10)
    boundaryPoints = textTiling.extractBoundaryPoints(lexicalScores, s=4, n=2)
    tiles = getTiles(article, boundaryPoints, w=20, k=10)
    tiledSummary = []
    for t in tiles:
        print("------")
        tiledSummary.append(textRankSummary(t, fraction))
    return tiledSummary, textRankSummary(article.getSentences(), fraction)
if __name__ == "__main__":
    article = preprocess.cachedArticle(argv[1])
    tiledSummary, simpleSummary = summarize(article, 0.1)
    print(simpleSummary)

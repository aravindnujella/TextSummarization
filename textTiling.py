from sys import argv
import preprocess
import similarity
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
import pdb
import pylab


def lowPassFilter(lexicalScores, s, n):
    smoothScores = list(lexicalScores)
    size = len(lexicalScores)
    for j in range(n):
        for i in range(size):
            leftSum = sum(smoothScores[int(max(0, i - s / 2)):i])
            rightSum = sum(smoothScores[min(i + 1, size):int(min(i + s / 2, size) + 1)])
            smoothScores[i] = (smoothScores[i] + leftSum + rightSum) / 3
    return smoothScores


def computeDepthScores(smoothScores):
    depthScores = []
    size = len(smoothScores)
    for i in range(size):
        l = i - 1
        r = i + 1
        while(l > 0 and smoothScores[l - 1] >= smoothScores[l]):
            l = l - 1
        while(r < size - 1 and smoothScores[r + 1] >= smoothScores[r]):
            r = r + 1
        depthScores.append((smoothScores[max(0, l)] - smoothScores[i]) + (smoothScores[min(r, size - 1)] - smoothScores[i]))
    return depthScores


def computeBoundaryPoints(depthScores, pts):
    boundaryPoints = []
    mean = np.mean(depthScores[1:-1])
    variance = np.var(depthScores[1:-1])
    for i in range(len(pts)):
        if(pts[i] >= (mean - (variance**0.5))):
            boundaryPoints.append(pts[i])
    return boundaryPoints


def extractBoundaryPoints(lexicalScores, s, n):
    smoothScores = lowPassFilter(lexicalScores, s, n)
    depthScores = lowPassFilter(computeDepthScores(smoothScores), s=2, n=1)
    pts = localMaxima(depthScores[1:-1])
    boundaryPoints = computeBoundaryPoints(depthScores, pts)

    return boundaryPoints


class boundary:

    def __init__(self, start, end, scores):
        self.start = start
        self.end = end
        self.scores = scores

    # Merge other on right...
    def merge(self, other):
        return boundary(self.start, other.end, self.scores + other.scores[1:])

    def cost(self):
        scores = self.scores
        # print(scores, self.start, self.end)
        return 2 * max(scores) - scores[-1] - scores[0]


def costOfMerge(left, right, lexicalScores):
    return left.cost() + right.cost() - left.merge(right).cost()


def localMinima(lexicalScores):
    l = []
    for i in range(1, len(lexicalScores) - 1):
        if lexicalScores[i] < lexicalScores[i + 1] and lexicalScores[i] <= lexicalScores[i - 1]:
            l.append(i)
    return l


def localMaxima(lexicalScores):
    l = []
    for i in range(1, len(lexicalScores) - 1):
        if lexicalScores[i] > lexicalScores[i + 1] and lexicalScores[i] >= lexicalScores[i - 1]:
            l.append(i)
    return l


def initialBoundaries(lexicalScores):
    start = 0
    while lexicalScores[start] > lexicalScores[start + 1]:
        start += 1
    end = len(lexicalScores) - 1
    while lexicalScores[end] > lexicalScores[end - 1]:
        end -= 1
    l = localMinima(lexicalScores)
    l = [l[i] for i in range(len(l)) if l[i] >= start and l[i] <= end]
    startOfBoundary = start
    boundaryPoints = []
    for i in l:
        boundaryPoints.append(boundary(startOfBoundary, i, lexicalScores[startOfBoundary:i + 1]))
        startOfBoundary = i
    return boundaryPoints


# TODO: Fixing number of clusters
def extractTiles1(article, lexicalScores, w, k):
    boundaryPoints = initialBoundaries(lexicalScores)
    windowWidth = w * k
    # Recursive(/Heirarchical) merge
    while True:
        # if some condition, break....
        minMerge = float('inf')
        mergeIndex = 0
        for i in range(len(boundaryPoints) - 1):
            c = costOfMerge(boundaryPoints[i], boundaryPoints[i + 1], lexicalScores)
            if c < minMerge:
                minMerge = c
                mergeIndex = i
        boundaryPoints = boundaryPoints[:mergeIndex] + [boundaryPoints[mergeIndex].merge(boundaryPoints[mergeIndex + 1])] + boundaryPoints[mergeIndex + 2:]
        # words = article.getWords()
        # sentenceID = article.getSentenceIDs()
        # f = plt.figure(0)
        # plt.plot([sentenceID[i] for i in range(windowWidth, len(words) - windowWidth + 1, w)], [lexicalScores[i] for i in range(len(lexicalScores))])
        # for b in boundaryPoints:
        #     plt.plot((sentenceID[w * k + w * b.start], sentenceID[w * k + w * b.start]), (0, 1), color='r')
        #     plt.plot((sentenceID[w * k + w * b.end], sentenceID[w * k + w * b.end]), (0, 1), color='r')
        # plt.show()


if __name__ == "__main__":
    article = preprocess.cachedArticle(argv[1])
    w = 20
    k = 10
    s = 4
    n = 2
    try:
        w = int(argv[2])
        k = int(argv[3])
        s = int(argv[4])
        n = int(argv[5])
    except:
        pass
    lexicalScores = similarity.computeSimilarity3(article, w, k)
    smoothScores = lowPassFilter(lexicalScores, s, n)
    # print(extractTiles1(article, lexicalScores, w, k))
    windowWidth = k * w
    depthScores = lowPassFilter(computeDepthScores(smoothScores), s=2, n=1)
    pts = localMaxima(depthScores[1:-1])

    boundaryPoints = computeBoundaryPoints(depthScores, pts)
    sentenceID = article.getSentenceIDs()
    words = article.getWords()

    sentences = article.getSentences()
    # print(boundaryPoints)
    # for i in boundaryPoints:
    #     print(sentences[sentenceID[w * k + w * i]])
    f = plt.figure(0)
    # plt.plot([sentenceID[i] for i in range(windowWidth, len(words) - windowWidth + 1, w)], [smoothScores[i] for i in range(len(smoothScores))])

    # for i in boundaryPoints:
    #     plt.plot((sentenceID[w * k + i * w], sentenceID[w * k + i * w]), (0, 0.2), 'r')
    # boundaryPoints = extractTiles(lexicalScores,2,1)
    st = 0
    for i in range(len(boundaryPoints)):
        print("=========Boundary " + str(i) + " =========")
        print(sentences[st : sentenceID[windowWidth + w * boundaryPoints[i]]])
        st = sentenceID[windowWidth + w * boundaryPoints[i]]
        print("=======================================")
    # print(sentences[st:])
    # pylab.plot(range(len(depthScores)), depthScores, color='r', label='DepthScores')
    # print(boundaryPoints)
    pylab.plot([sentenceID[i] for i in range(windowWidth, len(words) - windowWidth + 1, w)], [smoothScores[i] for i in range(len(smoothScores))])

    for b in boundaryPoints:
        pylab.plot((sentenceID[w * k + w * b], sentenceID[w * k + w * b]), (0, 1), color='r')

    # pylab.plot(range(len(smoothScores)), smoothScores, color='b', label='LexicalScores')
    pylab.legend(loc='upper left')
    plt.show()
    # input()

    # boundaryPoints = extractTiles1(smoothScores, article, w, k)

    # input()

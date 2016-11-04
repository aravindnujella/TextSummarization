from sys import argv
import preprocess
import similarity
import urllib.request
import numpy as np
import matplotlib.pyplot as plt


def lowPassFilter(lexicalScores, s, n):
    # copy the list
    smoothScores = list(lexicalScores)
    size = len(lexicalScores)
    # Average smoothing over a width of size s
    for j in range(n):
        for i in range(size):
            leftSum = sum(smoothScores[int(max(0, i - s / 2)):i])
            rightSum = sum(smoothScores[min(i + 1, size):int(min(i + s / 2, size) + 1)])
            smoothScores[i] = (smoothScores[i] + leftSum + rightSum) / 3
    return smoothScores


def extractTiles(lexicalScores, s, n):
    # First we have to smooth the lexical scores
    smoothScores = lowPassFilter(lexicalScores, s, n)
    size = len(smoothScores)
    f = plt.figure(1)
    # plt.plot(smoothScores)
    # f.show()
    depthScores = []
    # We assign a depthScore to every word
    for i in range(size):
        # find two indices l,r
        l = i - 1
        r = i + 1
        while(l > 0):
            if(smoothScores[l - 1] >= smoothScores[l]):
                l = l - 1
            else:
                break
        while(r < size - 1):
            if(smoothScores[r + 1] >= smoothScores[r]):
                r = r + 1
            else:
                break

        depthScores.append((smoothScores[max(0, l)] - smoothScores[i]) + (smoothScores[min(r, size - 1)] - smoothScores[i]))

    boundaryPoints = []
    mean = np.mean(depthScores)
    variance = np.var(depthScores)

    print(mean, variance)

    for i in range(size):
        if(depthScores[i] >= (mean + 3 * (variance**0.5))):
            boundaryPoints.append(i)
    # g = plt.figure(2)
    plt.plot(range(len(smoothScores)),smoothScores)
    plt.plot(range(len(depthScores)),depthScores,color='r')
    f.show()
    input()
    # We return the boundary points
    return boundaryPoints


def evaluationMeasure(text, boundaryPoints, w):
    # The boundary point is a set of 2 words.
    sentences = similarity.extractSentences(text)
    for i in range(len(boundaryPoints)):
        firstWord = w * (boundaryPoints[i])
        lastWord = w * (boundaryPoints[i] + 1)
        firstSentence = sentences[sentenceID[firstWord]]
        lastSentence = sentences[sentenceID[lastWord]]
        print(lastSentence, "\n")

if __name__ == "__main__":
    article = preprocess.article(argv[1])
    lexicalScores = similarity.computeSimilarity3(article.getText(), int(argv[2]), int(argv[3]))
    s = 4
    n = 2
    boundaryPoints = extractTiles(lexicalScores, s, n)
    words, sentenceID = similarity.extractWords(article.getText())
    evaluationMeasure(article.getText(), boundaryPoints, int(argv[2]))

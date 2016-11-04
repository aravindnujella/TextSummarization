from sys import argv
import preprocess
import similarity
import numpy as np
import matplotlib.pyplot as plt

# For Smoothing the graph
def lowPassFilter(lexicalScores, s, n):
    # Initialize
    smoothScores = list(lexicalScores)
    size = len(lexicalScores)
    # iteratively smoothing over a width of size s
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
def computeBoundaryPoints(depthScores):
    boundaryPoints = []
    mean = np.mean(depthScores)
    variance = np.var(depthScores)
    for i in range(len(depthScores)):
        if(depthScores[i] >= (mean + 3 * (variance**0.5))):
            boundaryPoints.append(i)
    return boundaryPoints    
def extractTiles(lexicalScores, s, n):
    smoothScores = lowPassFilter(lexicalScores, s, n)
    depthScores = computeDepthScores(smoothScores)
    boundaryPoints = computeBoundaryPoints(depthScores)
    plotGraph(smoothScores,depthScores)
    return boundaryPoints

def plotGraph(smoothScores,depthScores):
    f = plt.figure(0)
    plt.plot(range(len(smoothScores)), smoothScores)
    plt.plot(range(len(depthScores)), depthScores, color='r')
    f.show()
    input()
# def evaluationMeasure(article, boundaryPoints, w):
#     sentences = article.getSentences()
#     for i in range(len(boundaryPoints)):
#         firstWord = w * (boundaryPoints[i])
#         lastWord = w * (boundaryPoints[i] + 1)
#         firstSentence = sentences[sentenceID[firstWord]]
#         lastSentence = sentences[sentenceID[lastWord]]

if __name__ == "__main__":
    article = preprocess.cachedArticle(argv[1])
    w = 10
    k = 20
    try:
        w = int(argv[2])
        k = int(argv[3])
    except:
        pass

    lexicalScores = similarity.computeSimilarity3(article, w, k)
    boundaryPoints = extractTiles(lexicalScores, s=4, n=2)
    # evaluationMeasure(article, boundaryPoints, w)
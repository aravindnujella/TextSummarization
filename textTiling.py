from sys import argv
import preprocess
import similarity
import urllib.request


def extractTiles(lexicalScores):
    print("No extractTiles")
    return []
if __name__ == "__main__":
    article = preprocess.article(argv[1])
    lexicalScores = similarity.computeSimilarity(article.getText())
    extractTiles(lexicalScores)

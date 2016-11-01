from sys import argv
import preprocess
import similarity
import textTiling
import urllib.request

def textRankSummary(text):
    return "No textRankSummary"
if __name__ == "__main__":
    article = preprocess.article(argv[1])
    lexicalScores = similarity.computeSimilarity(article.getText())
    tiles = textTiling.extractTiles(lexicalScores)
    for t in tiles:
        textRankSummary(t)
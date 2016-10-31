from sys import argv
import preprocess
import similarity
import urllib.request


def extractTiles(lexicalScores):
    print("Lol")

if __name__ == "__main__":
    articleString = "https://en.wikipedia.org/w/api.php?format=json&action=query&prop=extracts&exlimit=max&explaintext&exsectionformat=plain&titles=" +argv[1] + "&redirects="
    response = urllib.request.urlopen(articleString)
    html = response.decode('utf-8')
    article = preprocess.article(html)
    lexicalScores = similarity.computeSimilarity(article.toText())
    extractTiles(lexicalScores)

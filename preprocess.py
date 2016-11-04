from sys import argv
import wikipedia
from unidecode import unidecode
import re


class article:

    def __init__(self, articleName):
        wikiPage = wikipedia.page(articleName)
        self.topics = {}
        self.text = ""
        # Cleaning, finding topics and plain text
        self.extract(wikiPage)

    def extract(self, wikiPage):
        # prelim cleaning, translitting non unicode stuff
        temp = unidecode(wikiPage.content)
        # remove all paranthesized stuff
        re.sub(r'\([^\)]*\)', '', temp)
        re.sub(r'\[[^\]]*\]', '', temp)
        topicsSearch = re.findall(r'=+[^=]*=+', temp)
        self.sections = []
        for t in re.findall(r'=+[^=]*=+', temp):
            match = re.search(r'=+([^=]*)=+', t)
            if match != None and match.group(1) != '':
                self.sections.append(match.group(1).strip())
        self.text += unidecode(wikiPage.summary).lower()
        for s in self.sections:
            if wikiPage.section(s) != None:
                self.topics[s] = unidecode(wikiPage.section(s)).lower()
                self.text += unidecode(wikiPage.section(s)).lower()

    def getText(self):
        return self.text

    def getTopics(self):
        return self.topics

class local_article:

    def __init__(self, articleName):
        wikiPage = wikipedia.page(articleName)
        self.topics = {}
        self.text = ""
        # Cleaning, finding topics and plain text
        self.extract(wikiPage)

    def extract(self, wikiPage):
        # prelim cleaning, translitting non unicode stuff
        temp = unidecode(wikiPage.content)
        # remove all paranthesized stuff
        re.sub(r'\([^\)]*\)', '', temp)
        re.sub(r'\[[^\]]*\]', '', temp)
        topicsSearch = re.findall(r'=+[^=]*=+', temp)
        self.sections = []
        for t in re.findall(r'=+[^=]*=+', temp):
            match = re.search(r'=+([^=]*)=+', t)
            if match != None and match.group(1) != '':
                self.sections.append(match.group(1).strip())
        self.text += unidecode(wikiPage.summary).lower()
        for s in self.sections:
            if wikiPage.section(s) != None:
                self.topics[s] = unidecode(wikiPage.section(s)).lower()
                self.text += unidecode(wikiPage.section(s)).lower()

    def getText(self):
        return self.text

    def getTopics(self):
        return self.topics

if __name__ == "__main__":
    someArticle = article(argv[1])
    print(someArticle.getText())
    print(someArticle.getTopics().keys())
    print(len(someArticle.getTopics().keys()))

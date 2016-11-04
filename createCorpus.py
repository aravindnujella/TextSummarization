import preprocess
from sys import argv
import pickle

for a in argv[1:]:
    article = preprocess.article(a)
    with open('./Corpus/' + a + '.pkl', 'wb') as f:
        pickle.dump(article,f)


import nltk
nltk.download('wordnet')

from nltk.corpus import wordnet

word = "car"
synlist = wordnet.synset(word)
print(synlist)

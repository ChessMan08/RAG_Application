from nltk.corpus import wordnet

def define(term: str) -> str:
    syns = wordnet.synsets(term)
    return syns[0].definition() if syns else "No definition found."

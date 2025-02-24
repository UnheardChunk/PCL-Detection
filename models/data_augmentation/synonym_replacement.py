import random
from nltk.corpus import wordnet

def get_synonyms(word):
    synonyms = set()
    
    for syn in wordnet.synsets(word): 
        for l in syn.lemmas(): 
            synonym = l.name().replace("_", " ").replace("-", " ").lower()
            synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
            synonyms.add(synonym) 
    
    if word in synonyms:
        synonyms.remove(word)
    
    return list(synonyms)

def synonym_replacement(text, num_words):
    words = text.split()
    
    if len(words) == 0:
        return text
    
    words_to_replace = random.sample(words, min(num_words, len(words)))
    
    new_words = words.copy()
    
    for word in words_to_replace:
        synonyms = get_synonyms(word)
        if synonyms:
            new_word = random.choice(synonyms)
            new_words = [new_word if w == word else w for w in new_words]
    
    return " ".join(new_words)
    
if __name__ == "__main__":
    text = "The quick brown fox jumps over the lazy dog"
    print(synonym_replacement(text, 3))

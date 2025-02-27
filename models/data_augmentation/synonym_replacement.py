import random
import nltk
import pandas as pd
from nltk.corpus import wordnet, stopwords

STOPWORDS = stopwords.words("english")

def get_wordnet_pos(tag):
    if tag.startswith("J"):
        return wordnet.ADJ
    elif tag.startswith("V"):
        return wordnet.VERB
    elif tag.startswith("N"):
        return wordnet.NOUN
    elif tag.startswith("R"):
        return wordnet.ADV
    else:
        return None

def get_synonyms(word, pos):
    synonyms = set()
    
    # Get synonyms from WordNet
    for syn in wordnet.synsets(word):
        # Only consider the synonyms of the same POS
        if syn.pos() != pos:
            continue
        
        for l in syn.lemmas():
            # If the synonym is not the same as the word then add it to the list
            if word.lower() != l.name().lower():
                synonym = l.name().replace("_", " ").lower()
                synonyms.add(synonym) 
    
    return list(synonyms)

def synonym_replacement(text, num_replacements):
    words = text.split()
    new_words = words.copy()
    # Get the POS tags for the words
    pos_tags = nltk.pos_tag(words)
    
    # If there are no words in the text then return the text
    if len(words) == 0:
        return text
    
    # Shuffle the indices of the words
    shuffled_indices = list(range(len(words)))
    random.shuffle(shuffled_indices)
    
    words_replaced = []
    word_pos_tags = []
    replaced = 0
    for i in shuffled_indices:
        # If the word is a stopword then skip it
        word = words[i]
        if word in STOPWORDS:
            continue
        
        # Get the POS tag of the word
        tag = pos_tags[i][1]
        wordnet_pos = get_wordnet_pos(tag)
        # If the wordnet_pos is None then skip the word
        if wordnet_pos is None:
            continue
        
        # Otherwise, get the synonyms of the word
        synonyms = get_synonyms(word, pos=wordnet_pos)
        if synonyms:
            # Replace the word with a random synonym
            synonym = random.choice(synonyms)
            new_words[i] = synonym
            words_replaced.append((word, synonym))
            word_pos_tags.append((word, tag))
            replaced += 1
        
        # If the number of words replaced is equal to the number of words to replace then break
        if replaced >= num_replacements:
            break
    
    return " ".join(new_words), words_replaced, word_pos_tags

def augment_dataset(dataset, num_replacements=3):
    augmented_rows = []
    for _, row in dataset.iterrows():
        augmented_text, _, _ = synonym_replacement(row["text"], num_replacements)
        augmented_rows.append({"text": augmented_text, "label": row["label"]})
    
    augmented_df = pd.DataFrame(augmented_rows)
    return pd.concat([dataset, augmented_df], ignore_index=True)
    
if __name__ == "__main__":
    text = "The quick brown fox jumps over the lazy dog"
    new_sentence, words_replaced, word_pos_tags = synonym_replacement(text, 3)
    print(new_sentence)
    print(words_replaced)
    print(word_pos_tags)

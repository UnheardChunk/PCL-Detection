import re
import pandas as pd

# Uncomment below when running for the first time to download data
# import nltk
# nltk.download('stopwords',download_dir="../../nlpenv/nltk_data")
# nltk.download('punkt_tab',download_dir="../../nlpenv/nltk_data")

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy

nlp = spacy.load("en_core_web_sm")


def remove_stop_words(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)

    filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
    return " ".join(filtered_sentence)


def remove_punctuation(text):
    return re.sub(r"[^\w\s]|_", "", text)
    

def lemmatize_text(text):
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc])
    

def preprocess_dataset(dataset, remove_punc=True, remove_stopwords=True, lemmatize=True):
    
    if remove_punc:
        # Remove punctuation from the text
        dataset["text"] = dataset["text"].apply(remove_punctuation)

    if remove_stopwords:
        # Remove stop words from text
        dataset["text"] = dataset["text"].apply(remove_stop_words)
    
    if lemmatize:
        # Lemmatize text
        dataset["text"] = dataset["text"].apply(lemmatize_text)

    return dataset


if __name__ == "__main__":
    text = "Hello, world!@£$%^&*()_+{}[]:;\"'<>,.?/~`-="
    print(text)
    print(remove_punctuation(text))
    print()

    text3 = "This is a sample sentence, showing off the stop words filtration."
    print(text3)
    print(remove_stop_words(text3))
    print()

    text2 = "The cats are running faster than the mice"
    print(text2)
    print(lemmatize_text(text2))
    print()

    df = pd.DataFrame({"text": [text, text3, text2]})
    print(preprocess_dataset(df))

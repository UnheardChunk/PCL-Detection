import re
import pandas as pd

# Uncomment below when running for the first time to download data
#import nltk
#nltk.download('stopwords',download_dir="../../nlpenv/nltk_data")
#nltk.download('punkt_tab',download_dir="../../nlpenv/nltk_data")

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def remove_stop_words(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    
    filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
    return " ".join(filtered_sentence)

def remove_punctuation(text):
    return re.sub(r"[^\w\s]|_", "", text)

def preprocess_dataset(dataset):
    # Remove punctuation from the text
    dataset["text"] = dataset["text"].apply(remove_punctuation)

    # Remove stop words from text
    dataset["text"] = dataset["text"].apply(remove_stop_words)
    
    # Apply any other preprocessing steps here
    # ...
    
    return dataset

if __name__ == "__main__":
    text = "Hello, world!@£$%^&*()_+{}[]:;\"'<>,.?/~`-="
    print(remove_punctuation(text))
    print()

    example_sent = "This is a sample sentence, showing off the stop words filtration."
    print(example_sent)
    print(remove_stop_words(example_sent))
    print()
    
    df = pd.DataFrame({"text": [text, example_sent]})
    print(preprocess_dataset(df))
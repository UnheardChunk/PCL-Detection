import re

def remove_punctuation(text):
    return re.sub(r"[^\w\s]|_", "", text)

def preprocess_dataset(dataset):
    # Remove punctuation from the text
    dataset["text"] = dataset["text"].apply(remove_punctuation)
    
    # Apply any other preprocessing steps here
    # ...
    
    return dataset

if __name__ == "__main__":
    text = "Hello, world!@£$%^&*()_+{}[]:;\"'<>,.?/~`-="
    print(remove_punctuation(text))
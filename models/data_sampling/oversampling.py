from imblearn.over_sampling import ADASYN
from sklearn.feature_extraction.text import TfidfVectorizer

import pandas as pd

# Oversample the minority class by randomly duplicating samples from the minority class
def random_oversample(dataset):
    # Get the minority class
    minority_class = dataset[dataset["label"] == 1]
    # Get the majority class
    majority_class = dataset[dataset["label"] == 0]
    
    # Randomly sample the minority class
    additional_samples_needed = len(majority_class) - len(minority_class)
    random_majority_samples = minority_class.sample(n=additional_samples_needed, replace=True, random_state=42)
    
    # Concatenate the original dataset with the randomly sampled minority class
    return pd.concat([dataset, random_majority_samples], ignore_index=True)    

# Oversample the minority class using the ADASYN algorithm
def oversample_adasyn(dataset):
    X = dataset["text"]
    y = dataset["label"]
    
    # Vectorize text data
    vectorizer = TfidfVectorizer()
    X_vectorized = vectorizer.fit_transform(X)
    
    # Initialize the ADASYN oversampler and resample the data
    ada = ADASYN(random_state=42)
    X_resampled, y_resampled = ada.fit_resample(X_vectorized, y)
    
    # Inverse the vectorised data back into text
    texts_resampled = vectorizer.inverse_transform(X_resampled)
    texts_resampled = [" ".join(text) for text in texts_resampled]

    return pd.DataFrame({"text": texts_resampled, "label": y_resampled})


if __name__ == "__main__":
    dataset_path = "./dataset/original_datasets/dontpatronizeme_pcl.tsv"

    dataset = pd.read_csv(dataset_path, sep="\t", skiprows=4, names=['par_id', 'art_id', 'keyword', 'country', 'text', 'orig_label'], index_col=0)
    dataset.head()
    dataset.loc[dataset["text"].isna(), "text"] = ""
    dataset["label"] = dataset["orig_label"].apply(lambda x : 0 if (x == 0 or x == 1) else 1)
    
    # balanced_dataset = oversample_adasyn(dataset)
    balanced_dataset = random_oversample(dataset)
    
    print("Class distribution before oversampling:")
    print(dataset["label"].value_counts())
    
    print("Class distribution after oversampling:")
    print(balanced_dataset["label"].value_counts())
    
    # Display a sample of old minority class
    print(balanced_dataset.head())
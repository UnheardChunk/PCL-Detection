from imblearn.under_sampling import ClusterCentroids
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

dataset_path = "../../dataset/original_datasets/dontpatronizeme_pcl.tsv"

dataset = pd.read_csv(dataset_path, sep="\t", skiprows=4, names=['par_id', 'art_id', 'keyword', 'country', 'text', 'orig_label'], index_col=0)
dataset.head()
dataset.loc[dataset["text"].isna(), "text"] = ""
dataset["label"] = dataset["orig_label"].apply(lambda x : 0 if (x == 0 or x == 1) else 1)

def undersample(df):

    X = df['text']
    y = df['label']
    
    # Vectorize text data
    vectorizer = TfidfVectorizer()
    X_vectorized = vectorizer.fit_transform(X)
    
    # Initialize the ClusterCentroids undersampler
    cc = ClusterCentroids(random_state=42)
    X_resampled, y_resampled = cc.fit_resample(X_vectorized, y)

    df_resampled = pd.DataFrame(X_resampled.toarray(), columns=vectorizer.get_feature_names_out())
    df_resampled['label'] = y_resampled

    return df_resampled

# Display the class distribution after undersampling
print(dataset['label'].value_counts())
print("After undersampling\n")
print(undersample(dataset)['label'].value_counts())


import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler, TensorDataset
import pandas as pd
import torch
from transformers import AutoTokenizer, DataCollatorWithPadding
from datasets import Dataset

def class_weighting(df):
   
    labels = np.array(df['label'])
    print("Labels:", str(labels))
    
    # Count occurrences of each class
    class_counts = np.bincount(labels)
    print("Class counts:", class_counts)
    
    # Compute inverse frequency weights
    class_weights = 1.0 / class_counts  # Smaller class = higher weight
    print("Class Weights:", class_weights)
    
    # Assign a weight to each sample
    # Map each label to its corresponding class weight
    sample_weights = class_weights[labels]
    
    # Define a sampler using these weights
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    return sampler


if __name__ == "__main__":
    dataset_path = "../../dataset/original_datasets/dontpatronizeme_pcl.tsv"

    dataset = pd.read_csv(dataset_path, sep="\t", skiprows=4, names=['par_id', 'art_id', 'keyword', 'country', 'text', 'orig_label'], index_col=0)
    dataset.head()
    dataset.loc[dataset["text"].isna(), "text"] = ""
    dataset["label"] = dataset["orig_label"].apply(lambda x : 0 if (x == 0 or x == 1) else 1)

    print("Class Distribution Before Sampling:\n", dataset['label'].value_counts())

    # Generate sampler
    sampler = class_weighting(dataset)

    # Tokenize dataset
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-small", model_max_length=512)
    
    def tokenize_function(example):
        return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)

    dataset_hf = Dataset.from_pandas(dataset[['text', 'label']])
    tokenized_dataset = dataset_hf.map(tokenize_function, batched=True, remove_columns=['text'])

    # Convert tokenized dataset to PyTorch tensors
    input_ids = torch.tensor(tokenized_dataset['input_ids'])
    attention_mask = torch.tensor(tokenized_dataset['attention_mask'])
    labels = torch.tensor(tokenized_dataset['label'])

    tensor_dataset = TensorDataset(input_ids, attention_mask, labels)

    # Create DataLoader with sampler
    train_loader = DataLoader(tensor_dataset, batch_size=16, sampler=sampler)
    num_samples = train_loader.sampler.num_samples
    print("Number of samples used by the sampler:", num_samples)

    # Check some sampled data
    for batch in train_loader:
        print("Sampled batch:", batch)
        break  # Print one batch and exit
    

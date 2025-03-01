import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
import pandas as pd
from transformers import AutoTokenizer, DataCollatorWithPadding
import pandas as pd
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import AutoTokenizer, DataCollatorWithPadding


checkpoint = "microsoft/deberta-v3-small"

# Perform tokenization
tokenizer = AutoTokenizer.from_pretrained(checkpoint, model_max_length=512)


def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)


# Example: Assume we have an imbalanced dataset with these labels
# Example dataset with 3 classes (imbalanced)
labels = np.array([0, 0, 1, 1, 1, 2, 2, 2, 2, 2])

# Count occurrences of each class
# e.g., [2, 3, 5] (how many samples per class)
class_counts = np.bincount(labels)

# Compute inverse frequency weights
class_weights = 1.0 / class_counts  # Smaller class = higher weight
print("Class Weights:", class_weights)

# Assign a weight to each sample
# Map each label to its corresponding class weight
sample_weights = class_weights[labels]

# Define a sampler using these weights
sampler = WeightedRandomSampler(
    weights=sample_weights, num_samples=len(sample_weights), replacement=True)

train_dataset_path = "../dataset/dpm_pcl_train.csv"
train_dataset = pd.read_csv(train_dataset_path)

val_dataset_path = "../dataset/dpm_pcl_val.csv"
val_dataset = pd.read_csv(val_dataset_path)

test_dataset_path = "../dataset/dpm_pcl_test.csv"
test_dataset = pd.read_csv(test_dataset_path)

train_dataset.head()

train_dataset["label"] = train_dataset["orig_label"].apply(
    lambda x: 0 if (x == 0 or x == 1) else 1)
val_dataset["label"] = val_dataset["orig_label"].apply(
    lambda x: 0 if (x == 0 or x == 1) else 1)
test_dataset["label"] = test_dataset["orig_label"].apply(
    lambda x: 0 if (x == 0 or x == 1) else 1)

train_dataset.loc[train_dataset["text"].isna(), "text"] = ""
val_dataset.loc[val_dataset["text"].isna(), "text"] = ""
test_dataset.loc[test_dataset["text"].isna(), "text"] = ""

set_uncased = False
if set_uncased:
    train_dataset['text'] = train_dataset['text'].str.lower()
    val_dataset['text'] = val_dataset['text'].str.lower()
    test_dataset['text'] = test_dataset['text'].str.lower()

train_dataset = train_dataset.drop(
    ['par_id', 'art_id', 'keyword', 'country', 'orig_label'], axis=1)
val_dataset = val_dataset.drop(
    ['par_id', 'art_id', 'keyword', 'country', 'orig_label'], axis=1)
test_dataset = test_dataset.drop(
    ['par_id', 'art_id', 'keyword', 'country', 'orig_label'], axis=1)

train_dataset.head()


raw_datasets_train = Dataset.from_pandas(train_dataset[['text', 'label']])

tokenized_datasets_train = raw_datasets_train.map(
    tokenize_function, batched=True, remove_columns=['text'])

# DataCollatorWithPadding constructs batches that are padded to the length of the longest sentence in the batch
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Example: Applying this to a PyTorch DataLoader
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=16, sampler=sampler, collate_fn=data_collator)

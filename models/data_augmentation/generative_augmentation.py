from transformers import pipeline, AutoTokenizer
from tqdm import tqdm

import pandas as pd

model_name = "ramsrigouthamg/t5_paraphraser"
paraphrase_pipe = pipeline("text2text-generation", model=model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def paraphrase(texts, max_length=150):
    outputs = paraphrase_pipe(
        texts, 
        max_length=max_length, 
        num_return_sequences=1, 
        do_sample=True,  # Enables more variation in output
        top_k=50,  # Controls randomness
        top_p=0.95,  # Nucleus sampling
        batch_size=len(texts)
    )
    
    paraphrased_texts = []
    for output in outputs:
        paraphrased_texts.append(output["generated_text"])
    
    return paraphrased_texts

def augment_dataset(dataset, batch_size=32):
    texts = dataset["text"].tolist()
    augmented_texts = []
    for i in tqdm(range(0, len(dataset), batch_size)):
        batch_texts = texts[i:i+batch_size]
        
        # Prepend the paraphrase prompt to each text
        batch_input = [f"Paraphrase: {text}" for text in batch_texts]
        
        # Get the maximum length of the batch texts
        token_lengths = [len(tokenizer.encode(text, truncation=True)) for text in batch_input]
        max_token_length = max(token_lengths) if token_lengths else 0
        
        # Determine paraphrase output token length: at least 25 tokens or half of the maximum token length
        # but not exceeding 512 tokens
        output_max_length = min(max(25, max_token_length // 2), 512)
        
        # Paraphrase the batch
        augmented_texts += paraphrase(batch_input, max_length=output_max_length)
    
    # Create a copy of the original dataset and replace the 'text' column with the augmented texts
    augmented_dataset = dataset.copy()
    augmented_dataset["text"] = augmented_texts
    
    # Concatenate the original and augmented datasets
    return pd.concat([dataset, augmented_dataset], ignore_index=True)

if __name__ == "__main__":
    print()
    text = "this is a paraphrased sentence"
    print(text)
    print("after paraphrasing: ")
    print(paraphrase([text]))
    print()

    text = "It is miserable being poor and homeless in an affluent nation where one 's worth is measured by what one can afford to buy . Even those in the middle class with good and steady jobs struggle to keep their lifestyle ."
    print(text)
    print("after paraphrasing: ")
    print(paraphrase([text], max_length=len(text)))
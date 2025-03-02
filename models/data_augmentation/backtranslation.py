from concurrent.futures import ThreadPoolExecutor, as_completed
from deep_translator import GoogleTranslator, MyMemoryTranslator
from tqdm import tqdm

import pandas as pd
import time

def backtranslate_batch(texts):
    count = 0
    while count < 3:
        try:
            # Throttle slightly before making requests
            time.sleep(1.2)
            translated = GoogleTranslator(source="en", target="japanese").translate_batch(texts)
            time.sleep(1.2)
            return GoogleTranslator(source="japanese", target="en").translate_batch(translated)
        except Exception as e:
            time.sleep(5)
            count += 1
            print(f"Failed to backtranslate batch on attempt {count + 1}: {e}")
    
    print(f"Failed to backtranslate batch after 3 attempts")
    return texts

def backtranslate_parallel(texts, batch_size=200, max_workers=5):
    # Partition texts into batches
    batches = [texts[i:i+batch_size] for i in range(0, len(texts), batch_size)]
    # Initialize list to store results, so that we can preserve the order of the texts
    results_batches = [None] * len(batches)
    
    # Process batches concurrently using ThreadPoolExecutor
    # Set max_workers to 5 to avoid hitting the request limits of the translation APIs
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(backtranslate_batch, batch): idx for idx, batch in enumerate(batches)}
        for future in tqdm(as_completed(futures), total=len(batches), desc="Backtranslating"):
            idx = futures[future]
            try:
                results_batches[idx] = future.result()
            except Exception as e:
                print(f"Error processing batch {idx}: {e}")
                # Fallback to original texts
                results_batches[idx] = batches[idx]
    
    # Flatten the results
    return [text for batch in results_batches for text in batch]

def augment_dataset(dataset):
    texts = dataset['text'].tolist()
    
    # Generate augmented texts by backtranslating the original texts
    aug_texts = backtranslate_parallel(texts)
    
    # Create a copy of the original dataframe and replace the 'text' column with the augmented texts
    augmented_dataset = dataset.copy()
    augmented_dataset['text'] = aug_texts
    
    # Concatenate the original and augmented datasets
    augmented_dataset = pd.concat([dataset, augmented_dataset], ignore_index=True)
    
    # Remove any duplicate rows
    return augmented_dataset.drop_duplicates(subset=["text"], ignore_index=True)

if __name__ == "__main__":
    texts = [
        "The governor said the funds would be channeled to bursaries and scholarships for bright children from poor families .",
        "It is miserable being poor and homeless in an affluent nation where one 's worth is measured by what one can afford to buy . Even those in the middle class with good and steady jobs struggle to keep their lifestyle .",
        "In Libya today , there are countless number of Ghanaian and Nigerian immigrants . These are the two countries with key macroeconomic challenges including unemployment . Let 's tackle this issue from the root and not the fruit . Thank you",
        "Council customers only signs would be displayed . Two of the spaces would be reserved for disabled persons and there would be five P30 spaces and eight P60 ones .",
        "To bring down high blood sugar levels , insulin needs to be taken . If you are the type who requires insulin during meal time , you will have to take correct doses of insulin in order to lower your blood glucose . Decision needs to be taken on when to inject it and how many times to inject . For this , you have to take the help of a health care professional .",
        "The European Union is making an historic mistake in its haste to conclude a refugee deal with Turkey , overlooking human rights violations that risk plunging the bloc 's largest membership candidate into civil war , said Selahattin Demirtas , leader of the nation 's most prominent pro-Kurdish party .",
        "NUEVA ERA , Ilocos Norte - No family shall be homeless under the watch of the municipal government here , said town Mayor Aldrin Garvida .",
        "His spokesman said the Kremlin needed more information about the rebels ' decision . He also said the rebel statement came only after the Western-backed government in Kiev had declared it would press on with its military operation , implying that Ukraine was to blame for the rebels ' refusal to heed Putin .",
        "A federal appeals court on Tuesday cleared the way for a 17-year-old immigrant held in custody in Texas to obtain an abortion . The full US Court of Appeals for the District of Columbia Circuit ruled 6-3 in favor of the teen . The decision overturned a ruling by a three-judge panel of the court that at least temporarily blocked her from getting an abortion . The Trump administration could still appeal the decision to the Supreme Court .",
        "We find ourselves in this situation because people are living longer and a number of more local factors including that the number of people requiring nursing care in their care home is increasing coupled with the increased demands for council-funded care to vulnerable people within their own homes .",
        "In the Government 's commitment to protect vulnerable groups , Najib in the 2017 Budget proposed financial assistance to poor families including General Assistance up to RM300 per month and Children Assistance up to RM450 per month ."
    ]
    dataset = pd.DataFrame({"text": texts})
    
    translated = augment_dataset(dataset)
    
    print(translated)
    
    texts = dataset['text'].tolist()
    translated_texts = translated['text'].tolist()
    for i in range(len(texts)):
        print(f"Original: {texts[i]}")
        print(f"Backtranslated: {translated_texts[i + len(texts)]}\n")
        print()

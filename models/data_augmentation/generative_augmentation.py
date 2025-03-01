from transformers import pipeline

paraphrase_pipe = pipeline("text2text-generation",
                           model="ramsrigouthamg/t5_paraphraser")

def paraphrase(text, max_length=150):
    return paraphrase_pipe(f"Paraphrase: {text}", 
                           max_length=max_length, 
                           num_return_sequences=1, 
                           do_sample=True,  # Enables more variation in output
                           top_k=50,  # Controls randomness
                           top_p=0.95  # Nucleus sampling
                           )[0]["generated_text"]


print()
text = "this is a paraphrased sentence"
print(text)
print("after paraphrasing: ")
print(paraphrase(text))
print()

text = "It is miserable being poor and homeless in an affluent nation where one 's worth is measured by what one can afford to buy . Even those in the middle class with good and steady jobs struggle to keep their lifestyle ."
print(text)
print("after paraphrasing: ")
print(paraphrase(text, max_length=len(text)))
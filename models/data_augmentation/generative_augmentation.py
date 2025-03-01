from transformers import pipeline

paraphrase_pipe = pipeline("text2text-generation",
                           model="ramsrigouthamg/t5_paraphraser")


def paraphrase(text):
    return paraphrase_pipe(f"Paraphrase: {text}")[0]["generated_text"]


print(paraphrase("this is a paraphrased sentence"))

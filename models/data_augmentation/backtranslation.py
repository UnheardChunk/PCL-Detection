from deep_translator import GoogleTranslator

def backtranslate(text):
    translated = GoogleTranslator(source="en", target="japanese").translate(text)
    back_translated = GoogleTranslator(source="japanese", target="en").translate(translated)
    return back_translated

if __name__ == "__main__":
    text = "It is miserable being poor and homeless in an affluent nation where one 's worth is measured by what one can afford to buy . Even those in the middle class with good and steady jobs struggle to keep their lifestyle ."
    print("Original:", text)
    print("Back Translated:", backtranslate(text))

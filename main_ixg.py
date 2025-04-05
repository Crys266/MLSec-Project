from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sc_ixgù import IXGSequenceClassificationExplainer
from transformers_interpret import SequenceClassificationExplainer

# Carica il modello e il tokenizer
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

print(f"Numero di classi: {model.config.num_labels}")
print("Label del modello:", model.config.id2label)

# Inizializza l'explainer
explainer_ixg = IXGSequenceClassificationExplainer(model, tokenizer)
explainer_lig = SequenceClassificationExplainer(model, tokenizer)

# Testa con un testo di esempio
text = "I love using transformers for natural language processing task."
attributions_ixg = explainer_ixg(text)
attributions_lig = explainer_lig(text)

# Stampa le attribuzioni
print("Word Attributions IxG:")
for word, score in attributions_ixg:
    print(f"{word}: {score}")

print("Word Attributions LIG:")
for word, score in attributions_lig:
    print(f"{word}: {score}")
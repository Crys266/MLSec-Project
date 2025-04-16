import torch
import torch.nn.functional as F
from transformers import BertForSequenceClassification, BertTokenizer
from transformers_interpret import SequenceClassificationExplainer
from captum.metrics import infidelity

# 1. Carica modello e tokenizer
model = BertForSequenceClassification.from_pretrained("textattack/bert-base-uncased-imdb")
model.eval()
tokenizer = BertTokenizer.from_pretrained("textattack/bert-base-uncased-imdb")

text = "I love this fantasy film."

# 2. Tokenizza il testo e ottieni gli embeddings
inputs = tokenizer(text, return_tensors="pt", truncation=True)
# Ottieni gli embeddings (input continui) direttamente dalla parte di embedding del modello:
embeddings = model.bert.embeddings(inputs["input_ids"])  # shape: [1, L_embed, D]

# 3. Calcola la predizione
with torch.no_grad():
    logits = model(**inputs).logits
    probs = F.softmax(logits, dim=-1)
pred_class_idx = torch.argmax(probs, dim=1).item()

# 4. Genera la spiegazione con transformers-interpret
explainer = SequenceClassificationExplainer(model, tokenizer)
# La spiegazione è una lista di tuple (token, attribution_scalar)
explanation = explainer(text, index=pred_class_idx)
print(f"\nWord attributions for: \"{text}\"")
for token, score in explanation:
    print(f"{token:>10} : {score:.4f}")
# Estrai i token aggregati e le rispettive attribuzioni
aggregated_tokens = [item[0] for item in explanation]       # Es.: ["[CLS]", "i", "love", "this", "fantasy", "film", ".", "[SEP]"]
aggregated_attributions = [float(item[1]) for item in explanation]

# 5. Confronta la tokenizzazione "completa" con quella aggregata
full_subtokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
print("Tokenizzazione completa:", full_subtokens)
print("Token aggregati dalla spiegazione:", aggregated_tokens)
# Supponiamo, ad esempio, che la tokenizzazione completa generi più sottotoken:
# Potresti ottenere, ad esempio: ['[CLS]', 'i', 'love', 'this', 'fan', '##tas', '##y', 'film', '.', '[SEP]']
# Noterai che "fantasy" compare come "fan", "##tas", "##y" (tre sottotoken)
# mentre aggregated_tokens potrebbe avere l'elemento "fantasy" con un'unica attribuzione.

# 6. Espandi le attribution in base al numero di sottotoken per ciascun token aggregato.
expanded_attributions = []
for token, att in zip(aggregated_tokens, aggregated_attributions):
    # Per token speciali come [CLS] o [SEP], assumiamo che non ci siano suddivisioni
    if token in ["[CLS]", "[SEP]"]:
        count = 1
    else:
        # Usa il tokenizer per ottenere i sottotoken
        subwords = tokenizer.tokenize(token)
        count = len(subwords)
    expanded_attributions.extend([att] * count)

# Verifica che la lunghezza delle attribution espanse corrisponda a quella della sequenza degli embeddings
if len(expanded_attributions) != embeddings.shape[1]:
    raise ValueError(f"Mismatch: attribution espanse = {len(expanded_attributions)}, "
                     f"ma gli embeddings hanno {embeddings.shape[1]} token.")

# Convertilo in tensore e espandi lungo la dimensione delle features (hidden_dim)
expanded_attr_tensor = torch.tensor(expanded_attributions, device=embeddings.device).unsqueeze(0)  # [1, L_embed]
attributions_tensor = expanded_attr_tensor.unsqueeze(-1).expand_as(embeddings)  # [1, L_embed, D]

# 7. Definisci la forward function che lavora sugli embeddings.
def custom_forward(embeds, attention_mask=None):
    # Passa gli embeddings al modello tramite l'argomento inputs_embeds.
    outputs = model(inputs_embeds=embeds, attention_mask=attention_mask)
    logits = outputs.logits
    # Restituisce il logit per la classe predetta
    return logits[:, pred_class_idx]

# 8. Definisci una funzione di perturbazione che lavora sugli embeddings.
def perturb_func(embeds, baselines=None):
    noise = torch.normal(mean=0, std=0.1, size=embeds.shape, device=embeds.device)
    perturbed = embeds + noise
    return noise, perturbed

# 9. Prepara il baseline: usa uno zero embedding della stessa forma.
baseline = torch.zeros_like(embeddings)

# 10. Calcola l'infedeltà utilizzando la funzione infidelity di Captum
infid_score = infidelity(
    forward_func=custom_forward,
    perturb_func=perturb_func,
    inputs=embeddings,
    attributions=attributions_tensor,
    baselines=baseline,
    n_perturb_samples=10
)

print("Infidelity score:", infid_score)

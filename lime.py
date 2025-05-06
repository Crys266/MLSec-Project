import re
import numpy as np
import torch
import torch.nn.functional as F
from transformers import BertForSequenceClassification, BertTokenizer
from lime.lime_text import LimeTextExplainer

# Setup device and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertForSequenceClassification.from_pretrained("textattack/bert-base-uncased-imdb").to(device)
tokenizer = BertTokenizer.from_pretrained("textattack/bert-base-uncased-imdb")
model.eval()

# Class labels and example sentences
class_labels = ["Negative", "Positive"]
sentences = [
    "I really love watch fantasy films!",
    "The film was a complete waste of time.",
]

# Tokenization and prediction (for initial demonstration)
inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
inputs = {key: value.to(device) for key, value in inputs.items()}
outputs = model(**inputs)
logits = outputs.logits
probabilities = F.softmax(logits, dim=-1)
predicted_classes = torch.argmax(probabilities, dim=-1).tolist()

# Prediction function for LIME
def lime_predict(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    probs = F.softmax(outputs.logits, dim=-1).cpu().numpy()
    return probs

# Custom tokenization function using regex that extracts words and punctuation
def model_tokenizer(text):
    bert_tokens = tokenizer.tokenize(text)
    cleaned_tokens = [token[2:] if token.startswith("##") else token for token in bert_tokens]

    output_tokens = []
    curr_pos = 0
    lower_text = text.lower()

    for idx, token in enumerate(cleaned_tokens):
        if token == "[UNK]":
            if idx + 1 < len(cleaned_tokens):
                next_token = cleaned_tokens[idx + 1].lower()
                pattern = re.escape(next_token)
                match = re.search(pattern, lower_text[curr_pos:])
                if match:
                    start = match.start()
                    # Prendi tutto il testo da curr_pos fino all'inizio del prossimo match
                    unk_text = text[curr_pos: curr_pos + start]
                    if unk_text:
                        output_tokens.append(unk_text)
                        curr_pos += len(unk_text)
                    continue  # il prossimo token verrà matchato nel ciclo successivo
            # Se non trovi token dopo o match fallisce, prendi un carattere solo
            if curr_pos < len(text):
                output_tokens.append(text[curr_pos])
                curr_pos += 1
            else:
                output_tokens.append("[UNK]")
        else:
            pattern = re.escape(token.lower())
            match = re.search(pattern, lower_text[curr_pos:])
            if match:
                start, end = match.span()
                output_tokens.append(text[curr_pos + start: curr_pos + end])
                curr_pos += end
            else:
                output_tokens.append(token)

    return output_tokens


lime_explainer = LimeTextExplainer(
    class_names=class_labels,
    split_expression=model_tokenizer,
    bow=False,
    char_level=False,
    kernel_width=5,
    feature_selection='forward_selection',
    random_state=42
)

all_attributions = []

print("=== LIME Explanation (Modified with Normalization) ===")
for i, sentence in enumerate(sentences):
    predicted_class_index = predicted_classes[i]
    predicted_class_label = class_labels[predicted_class_index]
    prob = probabilities[i][predicted_class_index].item()

    # Genera la spiegazione LIME per il campione corrente
    explanation = lime_explainer.explain_instance(
        sentence,
        lime_predict,
        num_features=100,
        labels=[predicted_class_index],
        num_samples=2000
    )
    print("predicted_class_index: ", predicted_class_index)

    # Ottieni i token usando il model_tokenizer (che garantisce di avere sottostringhe nel testo)
    tokens = model_tokenizer(sentence)

    # Recupera le attribuzioni come dizionario per il label corrente
    lime_dict = dict(explanation.as_list(label=predicted_class_index))

    # Allinea i token aggiungendo manualmente i token speciali se necessario
    lime_aligned = [("[CLS]", 0.0)]
    for tok in tokens:
        # Se il token non è presente, viene considerato con importanza 0.0
        lime_aligned.append((tok, lime_dict.get(tok, 0.0)))
    lime_aligned.append(("[SEP]", 0.0))

    # Applica normalizzazione: calcola la norma sui token (escludendo [CLS] e [SEP])
    scores = np.array([score for tok, score in lime_aligned if tok not in ["[CLS]", "[SEP]"]])
    norm = np.linalg.norm(scores)
    normalized_aligned = []
    for tok, score in lime_aligned:
        # Evitiamo divisione per zero
        norm_score = score / norm if norm != 0 else score
        normalized_aligned.append((tok, norm_score))

    # Se classe negativa, inverti
    if predicted_class_index == 0:
        normalized_aligned_text = [(tok, -score) for tok, score in normalized_aligned]
    else:
        normalized_aligned_text = normalized_aligned

    # === Modifica explanation.local_exp con valori normalizzati ===
    normalized_scores = {tok: norm_score for tok, norm_score in normalized_aligned}

    new_local_exp = []
    for feature_idx, original_score in explanation.local_exp[predicted_class_index]:
        token = explanation.domain_mapper.indexed_string.inverse_vocab[feature_idx]
        normalized_score = normalized_scores.get(token, 0.0)
        if predicted_class_index == 0:
            normalized_score = -normalized_score
        new_local_exp.append((feature_idx, normalized_score))

    explanation.local_exp = {predicted_class_index: new_local_exp}

    all_attributions.append({
         "sentence": sentence,
         "explain_type": "lime",
         "attributions": normalized_aligned_text,
         "pred_prob": prob,
         "pred_class": predicted_class_label
    })

    # Stampa il risultato per debug
    print(f"\nWord attributions for: \"{sentence}\"")
    print(f"Attribution type: LIME (normalized) for predicted class → {predicted_class_label} (p = {prob:.4f})")
    for token, score in normalized_aligned_text:
         print(f"{token:>10} : {score:+.4f}")
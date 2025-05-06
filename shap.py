import torch
import torch.nn.functional as F
from transformers import BertForSequenceClassification, BertTokenizer
import shap
import numpy as np
import warnings

# Optionally suppress the weight initialization warning if desired.
warnings.filterwarnings("ignore", message="Some weights of BertForSequenceClassification were not initialized")

# Load model and tokenizer
model = BertForSequenceClassification.from_pretrained("textattack/bert-base-uncased-imdb")
model.eval()
tokenizer = BertTokenizer.from_pretrained("textattack/bert-base-uncased-imdb")


def predict_fn(texts):
    # Debug: print the structure and length of texts
    # Note: texts may be a list of strings or a list-of-lists, depending on SHAP's masker.
    # Uncomment the next line to debug:
    # print("predict_fn - type(texts[0]):", type(texts[0]), "len(texts):", len(texts))

    # Determine the expected number of output rows.
    if isinstance(texts[0], list):
        n_expected = len(texts[0])
    else:
        n_expected = len(texts)

    # Ensure texts is a list of strings if needed:
    if not isinstance(texts[0], str):
        # If texts is a list-of-lists, flatten each inner element.
        texts_flat = []
        for inner in texts:
            # Convert each element to string if not already.
            if isinstance(inner, list):
                texts_flat.extend([str(t) for t in inner])
            else:
                texts_flat.append(str(inner))
        texts = texts_flat
        n_expected = len(texts)
    else:
        texts = [str(t) if t != "" else " " for t in texts]

    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs).logits
        probs = F.softmax(outputs, dim=-1)
    outputs_np = probs.detach().cpu().numpy()
    # Debug print: remove in production
    # print("predict_fn - outputs_np.shape:", outputs_np.shape, "n_expected:", n_expected)

    # If the number of output rows is less than n_expected, repeat the outputs.
    if outputs_np.shape[0] < n_expected:
        repeat_factor = int(np.ceil(n_expected / outputs_np.shape[0]))
        outputs_np = np.repeat(outputs_np, repeat_factor, axis=0)
        outputs_np = outputs_np[:n_expected]
    return outputs_np


# Use SHAP's Text masker by providing the tokenizer instance.
masker = shap.maskers.Text(tokenizer, mask_token=tokenizer.mask_token)

shap_explainer = shap.Explainer(predict_fn, masker)

text = "that movie was good."
shap_values = shap_explainer([text])

# Print SHAP explanation
print("=== SHAP Explanation ===")
print("Input:", text)
tokens = shap_values.data[0]
# Extract SHAP values for the positive class (class 1); adjust index if needed.
shap_scores = shap_values.values[0][:, 1]
for token, score in zip(tokens, shap_scores):
    print(f"Token: {token:12s} - SHAP Score: {score:8.4f}")

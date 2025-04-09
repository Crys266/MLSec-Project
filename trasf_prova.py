import torch
import torch.nn.functional as F
from transformers import BertForSequenceClassification, BertTokenizer
from transformers_interpret import SequenceClassificationExplainer

# Load model and tokenizer (warnings about uninitialized weights can be ignored for testing)
model = BertForSequenceClassification.from_pretrained("textattack/bert-base-uncased-imdb")
model.eval()
tokenizer = BertTokenizer.from_pretrained("textattack/bert-base-uncased-imdb")

text = "I love pink pussy."

# Compute predicted class and confidence separately.
inputs = tokenizer(text, return_tensors="pt", truncation=True)
with torch.no_grad():
    logits = model(**inputs).logits
    probs = F.softmax(logits, dim=-1)
predicted_class_idx = torch.argmax(probs, dim=1).item()
predicted_confidence = probs[0, predicted_class_idx].item()

# Create the interpreter for token attribution.
lig_explainer = SequenceClassificationExplainer(model, tokenizer)
# We pass the predicted class index so the attribution matches the chosen class.
lig_interpretation = lig_explainer(text, index=predicted_class_idx)

# Print the raw output for debugging.
print("=== Transformers Interpreter (LIG) Raw Output ===")
print(lig_interpretation)

# Here we assume the raw output is a list of tuples representing token-level attributions.
# We print all tokens, including "[CLS]".
tokens = [item[0] for item in lig_interpretation]
word_attributions = [item[1] for item in lig_interpretation]

print("\n=== Transformers Interpreter (LIG) Explanation ===")
print("Input:", text)
print("Predicted Class:", predicted_class_idx, "Confidence:", predicted_confidence)
print("Token Attribution Scores:")
for token, score in zip(tokens, word_attributions):
    token_str = token if isinstance(token, str) else str(token)
    # We print every token, including [CLS] and [SEP].
    display_token = token_str if token_str.strip() != "" else "[PAD]"
    try:
        score_float = float(score)
    except Exception:
        score_float = 0.0
    print(f"Token: {display_token:12s} - Attribution: {score_float:8.4f}")
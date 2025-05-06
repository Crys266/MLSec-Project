import torch
import random
import numpy as np


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    from transformers import BertTokenizer, BertForSequenceClassification
    from sequence_explainer import NewSequenceClassificationExplainer

    set_seed(42)

    model = BertForSequenceClassification.from_pretrained("textattack/bert-base-uncased-imdb")
    tokenizer = BertTokenizer.from_pretrained("textattack/bert-base-uncased-imdb")
    model.eval()

    explain_types = ["ig", "lig", "gs"]
    sentence = "I really love fantasy films."

    for explain_type in explain_types:
        explainer = NewSequenceClassificationExplainer(model, tokenizer, attribution_type=explain_type)
        attributions = explainer(sentence)

        # Stampa delle attribuzioni per ogni token nella frase.
        print(f"\nWord attributions for: \"{sentence}\"")
        print(f"Attribution type: {explain_type}")
        for token, score in attributions:
            print(f"{token:>10} : {score:.4f}")

        # Valutazione delle prestazioni
        print(f"Convergence Delta for {explain_type}: {explainer.attributions.delta.item():.4f}")
        print(f"Sensitivity for {explain_type}: {explainer.attributions.sensitivity.item():.4f}")

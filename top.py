

if __name__ == "__main__":
    from transformers import BertTokenizer, BertForSequenceClassification
    from sequence_explainer import NewSequenceClassificationExplainer

    model = BertForSequenceClassification.from_pretrained("textattack/bert-base-uncased-imdb")
    tokenizer = BertTokenizer.from_pretrained("textattack/bert-base-uncased-imdb")
    model.eval()

    explain_types = ["ig", "gs", "lig"]
    sentence = "I really love fantasy films."

    for explain_type in explain_types:
        explainer = NewSequenceClassificationExplainer(model, tokenizer, attribution_type=explain_type)
        attributions = explainer(sentence)

        # Print the attributions for each token in the sentence.
        print(f"\nWord attributions for: \"{sentence}\"")
        print(f"Attribution type: {explain_type}")
        for token, score in attributions:
            print(f"{token:>10} : {score:.4f}")
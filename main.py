import shap
import torch
import transformers
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from captum.attr import LayerIntegratedGradients
from captum.insights import AttributionVisualizer, Batch
from captum.insights.attr_vis.features import TextFeature
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers_interpret import SequenceClassificationExplainer
from reportlab.lib.pagesizes import letter, landscape
from reportlab.pdfgen import canvas

# Load model and tokenizer
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Sample text inputs
texts = ["I absolutely loved this movie!", "The film was a complete waste of time."]
print(f"Text inputs: {texts}")

# Tokenization
inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
print(f"Tokenized inputs: {inputs}")

# 1. Black-box Explanation with SHAP
print("\nComputing black-box explanation...")
# Define a wrapper function for SHAP that includes attention_mask
def model_wrapper(input_ids):
    # Convert input_ids to tensor
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    # Create attention_mask for the input_ids
    attention_mask = (input_ids != tokenizer.pad_token_id).int()
    # Pass input_ids and attention_mask to the model
    outputs = model(input_ids, attention_mask=attention_mask)
    return outputs.logits.detach().numpy()
# Create SHAP explainer
explainer = shap.Explainer(model_wrapper, inputs["input_ids"].numpy())
shap_values = explainer(inputs["input_ids"].numpy())

# 2. White-box Explanation with transformers-interpret
print("\nComputing white-box explanation...")
explainer_white = SequenceClassificationExplainer(model, tokenizer)
whitebox_explanations = [explainer_white(text) for text in texts]

# 3. White-box Explanation with Captum
print("\nComputing Captum white-box explanation...")
lig = LayerIntegratedGradients(lambda input_ids, attention_mask: model(input_ids, attention_mask=attention_mask).logits,
                               model.bert.embeddings)
captum_scores = []
for text in texts:
    input_dict = tokenizer(text, return_tensors="pt")
    input_ids = input_dict["input_ids"]
    attention_mask = input_dict["attention_mask"]
    # Compute attributions
    attributions = lig.attribute(inputs=input_ids, target=1, additional_forward_args=(attention_mask,))
    captum_scores.append(attributions.sum(dim=2).squeeze().detach().numpy())

# Display Results
for i, text in enumerate(texts):
    print(f"\nText: {text}")
    print("SHAP Explanation:", shap_values[i].values)
    print("Transformers-Interpret Explanation:", whitebox_explanations[i])
    print("Captum Explanation:", captum_scores[i])


# 4. Visualization with Matplotlib/Seaborn
def plot_attributions(tokens, attributions, title):
    plt.figure(figsize=(10, 6))
    sns.barplot(x=attributions, y=tokens, hue=tokens, palette="viridis", legend=False)
    plt.title(title)
    plt.xlabel("Attribution Value")
    plt.ylabel("Tokens")
    plt.show()


# Plot Transformers-Interpret Attributions
for i, text in enumerate(texts):
    tokens = [token for token, _ in whitebox_explanations[i]]
    attributions = [attribution for _, attribution in whitebox_explanations[i]]
    plot_attributions(tokens, attributions, f"Transformers-Interpret Attributions for: {text}")

# Plot Captum Attributions
for i, text in enumerate(texts):
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][i].tolist())
    captum_attributions = captum_scores[i]
    # Ensure tokens and attributions have the same length
    if len(tokens) != len(captum_attributions):
        print(f"Warning: Tokens and attributions have different lengths for text: {text}")
        min_length = min(len(tokens), len(captum_attributions))
        tokens = tokens[:min_length]
        captum_attributions = captum_attributions[:min_length]

    plot_attributions(tokens, captum_attributions, f"Captum Attributions for: {text}")


# 5. Create a Comparison Table
def create_comparison_table(texts, shap_values, whitebox_explanations, captum_scores, tokenizer):
    comparison_data = []
    for i, text in enumerate(texts):
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][i].tolist())
        shap_attributions = shap_values[i].values.mean(axis=1)
        transformers_attributions = [attribution for _, attribution in whitebox_explanations[i]]
        captum_attributions = captum_scores[i]
        min_length = min(len(tokens), len(shap_attributions), len(transformers_attributions), len(captum_attributions))
        tokens = tokens[:min_length]
        shap_attributions = shap_attributions[:min_length]
        transformers_attributions = transformers_attributions[:min_length]
        captum_attributions = captum_attributions[:min_length]
        for j, token in enumerate(tokens):
            comparison_data.append({
                "Text": text,
                "Token": token,
                "SHAP Attribution": shap_attributions[j],
                "Transformers-Interpret Attribution": transformers_attributions[j],
                "Captum Attribution": captum_attributions[j]
            })
    return pd.DataFrame(comparison_data)

comparison_table = create_comparison_table(texts, shap_values, whitebox_explanations, captum_scores, tokenizer)
comparison_table.to_csv("comparison_table.csv", index=False)

def convert_csv_to_pdf(csv_filename, pdf_filename):
    df = pd.read_csv(csv_filename)
    c = canvas.Canvas(pdf_filename, pagesize=landscape(letter))
    width, height = landscape(letter)
    c.setFont("Helvetica", 8)
    x_offset = 40
    y_offset = height - 40
    line_height = 10
    for i, col in enumerate(df.columns):
        c.drawString(x_offset + i * 100, y_offset, col)
    y_offset -= line_height
    for index, row in df.iterrows():
        for i, col in enumerate(df.columns):
            c.drawString(x_offset + i * 100, y_offset, str(row[col])[:25])
        y_offset -= line_height
        if y_offset < 40:
            c.showPage()
            c.setFont("Helvetica", 8)
            y_offset = height - 40
    c.save()

convert_csv_to_pdf("comparison_table.csv", "comparison_table.pdf")
print("PDF saved as comparison_table.pdf")


# 6. Visualization with SHAP
shap.initjs()
for i, text in enumerate(texts):
    print(f"\nSHAP Visualization for: {text}")
    # Convert input_ids to tokens
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][i].tolist())
    # Remove special tokens and padding (e.g., [CLS], [SEP], [PAD])
    valid_indices = [idx for idx, token in enumerate(tokens) if token not in ["[CLS]", "[SEP]", "[PAD]"]]
    tokens = [tokens[idx] for idx in valid_indices]
    shap_values_filtered = shap_values[i].values[valid_indices]
    # Ensure tokens and SHAP values have the same length
    if len(tokens) != len(shap_values_filtered):
        print(f"Warning: Tokens and SHAP values have different lengths for text: {text}")
        min_length = min(len(tokens), len(shap_values_filtered))
        tokens = tokens[:min_length]
        shap_values_filtered = shap_values_filtered[:min_length]
    # Create a SHAP Explanation object with filtered tokens and values
    shap_values_text = shap.Explanation(
        values=shap_values_filtered,
        base_values=shap_values[i].base_values,
        data=tokens,  # Use filtered tokens
        feature_names=tokens  # Use filtered tokens as feature names
    )
    # Plot SHAP values
    shap.plots.text(shap_values_text)



# 7. Visualization with Captum Insights
def model_forward(input_ids, attention_mask):
    return model(input_ids, attention_mask=attention_mask).logits

# Create a visualizer
visualizer = AttributionVisualizer(
    models=[model],
    score_func=lambda x: x.argmax(dim=-1),  # Show predicted class
    classes=["Negative", "Slightly Negative", "Neutral", "Slightly Positive", "Positive"],  # Class names
    features=[
        TextFeature("Input Text", tokenizer=tokenizer)
    ],
    dataset=Batch(inputs["input_ids"], inputs["attention_mask"])
)

# Start the visualization server
print("\nStarting Captum Insights visualization server...")
visualizer.render()
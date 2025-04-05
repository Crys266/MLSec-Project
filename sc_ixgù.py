import warnings
from typing import Dict, List, Optional, Tuple, Union

import torch
from captum.attr import visualization as viz
from torch.nn.modules.sparse import Embedding
from transformers import PreTrainedModel, PreTrainedTokenizer

from transformers_interpret.errors import AttributionTypeNotSupportedError, InputIdsNotCalculatedError
from a_ixg import InputXGradientAttributions  # Il nostro modulo IXG

SUPPORTED_ATTRIBUTION_TYPES = ["ixg"]  # Ora supportiamo solo IXG

class IXGSequenceClassificationExplainer:
    """
    Explainer per modelli di SequenceClassification basati su IXG (Input×Gradient).
    La struttura segue quella originale (usata per LIG) adattata per usare il metodo IXG.
    """
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        attribution_type: str = "ixg",
        custom_labels: Optional[List[str]] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = model.device

        if attribution_type != "ixg":
            raise AttributionTypeNotSupportedError(
                f"Attribution type '{attribution_type}' is not supported. Supported types: ['ixg']"
            )
        self.attribution_type = attribution_type

        if custom_labels is not None:
            if len(custom_labels) != len(model.config.label2id):
                raise ValueError(
                    f"`custom_labels` size '{len(custom_labels)}' should match pretrained model's label2id size '{len(model.config.label2id)}'"
                )
            self.id2label, self.label2id = self._get_id2label_and_label2id_dict(custom_labels)
        else:
            self.label2id = model.config.label2id
            self.id2label = model.config.id2label

        self.attributions: Optional[InputXGradientAttributions] = None
        self.input_ids: torch.Tensor = torch.Tensor()
        self._single_node_output = False

        self.internal_batch_size = None
        self.n_steps = 50  # Non usato in IXG, ma lasciato per compatibilità

    @staticmethod
    def _get_id2label_and_label2id_dict(labels: List[str]) -> Tuple[Dict[int, str], Dict[str, int]]:
        id2label: Dict[int, str] = dict()
        label2id: Dict[str, int] = dict()
        for idx, label in enumerate(labels):
            id2label[idx] = label
            label2id[label] = idx
        return id2label, label2id

    def encode(self, text: str) -> list:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, input_ids: torch.Tensor) -> list:
        return self.tokenizer.convert_ids_to_tokens(input_ids[0])

    @property
    def predicted_class_index(self) -> int:
        if len(self.input_ids) > 0:
            preds = self.model(self.input_ids)[0]
            self.pred_class = torch.argmax(torch.softmax(preds, dim=1)[0]).cpu().detach().numpy()
            return self.pred_class
        else:
            raise InputIdsNotCalculatedError("input_ids have not been created yet.")

    @property
    def predicted_class_name(self):
        try:
            index = self.predicted_class_index
            return self.id2label[int(index)]
        except Exception:
            return self.predicted_class_index

    @property
    def word_attributions(self) -> list:
        if self.attributions is not None:
            return self.attributions.word_attributions
        else:
            raise ValueError("Attributions have not yet been calculated. Please call the explainer on text first.")

    def visualize(self, html_filepath: str = None, true_class: str = None):
        tokens = [token.replace("Ġ", "") for token in self.decode(self.input_ids)]
        attr_class = self.id2label.get(self.selected_index, str(self.selected_index))
        if self._single_node_output:
            if true_class is None:
                true_class = round(float(self.pred_probs))
            predicted_class = round(float(self.pred_probs))
            attr_class = round(float(self.pred_probs))
        else:
            if true_class is None:
                true_class = self.selected_index
            predicted_class = self.predicted_class_name
        score_viz = self.attributions.visualize_attributions(
            pred_prob=self.pred_probs,
            pred_class=predicted_class,
            true_class=true_class,
            attr_class=attr_class,
            all_tokens=tokens,
        )
        html = viz.visualize_text([score_viz])
        if html_filepath:
            if not html_filepath.endswith(".html"):
                html_filepath += ".html"
            with open(html_filepath, "w") as f:
                f.write(html.data)
        return html

    def _forward(self, inputs_embeds, attention_mask):
        outputs = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        logits = outputs.logits
        return logits.squeeze(-1) if logits.shape[1] == 1 else logits

    def _make_input_reference_pair(self, text: str):
        encoding = self.tokenizer(text, return_tensors="pt")
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        # For compatibility with original structure, anche se IXG non usa la baseline
        ref_input_ids = torch.full_like(input_ids, self.tokenizer.pad_token_id).to(self.device)
        return input_ids, ref_input_ids, attention_mask

    def _calculate_attributions(self, embeddings: Embedding, index: int = None, class_name: str = None):
        # Per IXG non usiamo le funzioni avanzate di coppia input/reference
        self.input_ids, _, self.attention_mask = self._make_input_reference_pair(self.text)
        if index is not None:
            self.selected_index = index
        elif class_name is not None:
            if class_name in self.label2id:
                self.selected_index = int(self.label2id[class_name])
            else:
                warnings.warn(f"'{class_name}' not in label2id keys. Defaulting to predicted index.")
                self.selected_index = int(self.predicted_class_index)
        else:
            self.selected_index = int(self.predicted_class_index)

        # Ottieni i token di riferimento
        reference_tokens = [token.replace("Ġ", "") for token in self.decode(self.input_ids)]
        # Qui creiamo l'istanza di IXGAttributions (dal nuovo modulo a_ixg)
        ixg_attr = InputXGradientAttributions(
            custom_forward=self._forward,
            embeddings=embeddings,
            tokens=reference_tokens,
            attention_mask=self.attention_mask,
            target=None if self._single_node_output else self.selected_index,
        )
        ixg_attr.summarize()
        self.attributions = ixg_attr

    def _run(self, text: str, index: int = None, class_name: str = None, embedding_type: int = None) -> list:
        # Seleziona il tipo di embeddings (qui semplifichiamo: usiamo word embeddings)
        embeddings = self.model.get_input_embeddings()(self.input_ids)
        self.text = text  # memorizza il testo per uso nelle funzioni di calcolo
        self._calculate_attributions(embeddings=embeddings, index=index, class_name=class_name)
        return self.word_attributions

    def __call__(self, text: str, index: int = None, class_name: str = None, embedding_type: int = 0,
                 internal_batch_size: int = None, n_steps: int = None) -> list:
        if n_steps:
            self.n_steps = n_steps
        if internal_batch_size:
            self.internal_batch_size = internal_batch_size

        # Prepara input, tokenizzazione ed embeddings
        input_ids, ref_input_ids, attention_mask = self._make_input_reference_pair(text)
        self.input_ids = input_ids  # serve per calcolare predicted_class_index, etc.
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        embeddings_layer = self.model.get_input_embeddings()
        embeddings = embeddings_layer(input_ids)

        with torch.no_grad():
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
            if logits.shape[1] == 1:
                pred_probs = torch.sigmoid(logits)[0]
                pred_class = int((pred_probs > 0.5).item())
                selected_target = None
                self._single_node_output = True
            else:
                pred_probs = torch.softmax(logits, dim=-1)[0]
                pred_class = torch.argmax(pred_probs).item()
                selected_target = index if index is not None else pred_class
                self._single_node_output = False

        print("\n===== DEBUGGING INFORMATION =====")
        print(f"Testo analizzato: '{text}'")
        print(f"Classe predetta: {self.id2label[pred_class]} ({pred_class})")
        if selected_target is not None:
            print(f"Classe target per attribuzioni: {self.id2label[selected_target]} ({selected_target})")
        else:
            print("Nessuna classe target richiesta (output binario)")
        print(f"Probabilità delle classi: {pred_probs.detach().cpu().numpy()}")
        print("=================================\n")

        # Imposta le proprietà utili
        self.pred_probs = pred_probs
        self.selected_index = pred_class

        # Calcola le attribuzioni tramite il nuovo modulo IXG
        ixg_attr = InputXGradientAttributions(
            custom_forward=self._forward,
            embeddings=embeddings,
            tokens=tokens,
            attention_mask=attention_mask,
            target=selected_target,
        )
        ixg_attr.summarize()
        self.attributions = ixg_attr

        return ixg_attr.word_attributions

    def __str__(self):
        s = f"{self.__class__.__name__}(\n"
        s += f"\tmodel={self.model.__class__.__name__},\n"
        s += f"\ttokenizer={self.tokenizer.__class__.__name__},\n"
        s += f"\tattribution_type='{self.attribution_type}',\n"
        s += ")"
        return s

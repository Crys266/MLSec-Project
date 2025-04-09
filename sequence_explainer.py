import warnings
import torch
import torch.nn as nn
from typing import Callable, List, Optional, Tuple, Union, Dict
from abc import ABC, abstractmethod, abstractproperty
from transformers import PreTrainedModel, PreTrainedTokenizer
from captum.attr import GradientShap, visualization as viz
from transformers_interpret import BaseExplainer
from transformers_interpret.errors import (
    AttributionTypeNotSupportedError,
    InputIdsNotCalculatedError,
    AttributionsNotCalculatedError
)
import inspect, re
from torch.nn.modules.sparse import Embedding
from attributions import GSAttributions, IGAttributions, LIGAttributions


SUPPORTED_ATTRIBUTION_TYPES = ["lig", "ig", "gs"]



class NewSequenceClassificationExplainer(BaseExplainer):
    def __init__(
            self,
            model: PreTrainedModel,
            tokenizer: PreTrainedTokenizer,
            attribution_type: str = "gs",
            custom_labels: Optional[List[str]] = None
    ):
        super().__init__(model, tokenizer)
        if attribution_type not in SUPPORTED_ATTRIBUTION_TYPES:
            raise AttributionTypeNotSupportedError(
                f"""Attribution type '{attribution_type}' is not supported.
                Supported types are {SUPPORTED_ATTRIBUTION_TYPES}"""
            )
        self.attribution_type = attribution_type

        if custom_labels is not None:
            if len(custom_labels) != len(model.config.label2id):
                raise ValueError(
                    f"""`custom_labels` size '{len(custom_labels)}' should match pretrained model's label2id size
                          '{len(model.config.label2id)}'"""
                )

            self.id2label, self.label2id = self._get_id2label_and_label2id_dict(custom_labels)
        else:
            self.label2id = model.config.label2id
            self.id2label = model.config.id2label
        self.attributions: Union[None, GSAttributions, IGAttributions, LIGAttributions] = None
        self.input_ids: torch.Tensor = torch.Tensor()
        self._single_node_output = False
        self.internal_batch_size = None
        self.n_steps = 50

    @staticmethod
    def _get_id2label_and_label2id_dict(
            labels: List[str],
    ) -> Tuple[Dict[int, str], Dict[str, int]]:
        id2label: Dict[int, str] = dict()
        label2id: Dict[str, int] = dict()
        for idx, label in enumerate(labels):
            id2label[idx] = label
            label2id[label] = idx

        return id2label, label2id

    def encode(self, text: str = None) -> list:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, input_ids: torch.Tensor) -> list:
        return self.tokenizer.convert_ids_to_tokens(input_ids[0])

    @property
    def predicted_class_index(self) -> int:
        if len(self.input_ids) > 0:
            preds = self.model(self.input_ids)[0]
            return torch.argmax(torch.softmax(preds, dim=1)[0]).cpu().detach().numpy()
        else:
            raise InputIdsNotCalculatedError("input_ids not calculated yet.")

    @property
    def predicted_class_name(self):
        "Returns predicted class name (str) for model with last calculated `input_ids`"
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
            raise ValueError("Attributions not calculated. Call the explainer first.")

    def visualize(self, html_filepath: str = None, true_class: str = None):
        """
        Visualizes word attributions. If in a notebook table will be displayed inline.

        Otherwise pass a valid path to `html_filepath` and the visualization will be saved
        as a html file.

        If the true class is known for the text that can be passed to `true_class`

        """
        tokens = [token.replace("Ġ", "") for token in self.decode(self.input_ids)]
        attr_class = self.id2label[self.selected_index]

        if self._single_node_output:
            if true_class is None:
                true_class = round(float(self.pred_probs))
            predicted_class = round(float(self.pred_probs))
            attr_class = round(float(self.pred_probs))

        else:
            if true_class is None:
                true_class = self.selected_index
            predicted_class = self.predicted_class_name

        score_viz = self.attributions.visualize_attributions(  # type: ignore
            self.pred_probs,
            predicted_class,
            true_class,
            attr_class,
            tokens,
        )
        html = viz.visualize_text([score_viz])

        if html_filepath:
            if not html_filepath.endswith(".html"):
                html_filepath = html_filepath + ".html"
            with open(html_filepath, "w") as html_file:
                html_file.write(html.data)
        return html

    def _forward(  # type: ignore
        self,
        input_ids: torch.Tensor,
        token_type_ids=None,
        position_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
    ):

        preds = self._get_preds(input_ids, token_type_ids, position_ids, attention_mask)
        preds = preds[0]

        # if it is a single output node
        if len(preds[0]) == 1:
            self._single_node_output = True
            self.pred_probs = torch.sigmoid(preds)[0][0]
            return torch.sigmoid(preds)[:, :]

        self.pred_probs = torch.softmax(preds, dim=1)[0][self.selected_index]
        return torch.softmax(preds, dim=1)[:, self.selected_index]

    def _forward_embs(  # type: ignore
        self,
        inputs_embeds: torch.Tensor,
        additional_forward_args: Optional[Dict[str, torch.Tensor]] = None,
    ):
        attention_mask = additional_forward_args.get("attention_mask") if additional_forward_args else None
        token_type_ids = additional_forward_args.get("token_type_ids") if additional_forward_args else None
        position_ids = additional_forward_args.get("position_ids") if additional_forward_args else None

        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

        if logits.shape[1] == 1:
            self._single_node_output = True
            self.pred_probs = torch.sigmoid(logits)[0][0]
            return torch.sigmoid(logits)

        self.pred_probs = torch.softmax(logits, dim=1)[0][self.selected_index]
        return logits  # no softmax here, Captum handles it

    def _calculate_attributions(self, embeddings: Embedding, index: int = None, class_name: str = None):
        (
            self.input_ids,
            self.ref_input_ids,
            self.sep_idx
        ) = self._make_input_reference_pair(self.text)

        (
            self.position_ids,
            self.ref_position_ids,
        ) = self._make_input_reference_position_id_pair(self.input_ids)

        (
            self.token_type_ids,
            self.ref_token_type_ids,
        ) = self._make_input_reference_token_type_pair(self.input_ids, self.sep_idx)

        self.attention_mask = self._make_attention_mask(self.input_ids)

        if index is not None:
            self.selected_index = index
        elif class_name is not None:
            if class_name in self.label2id.keys():
                self.selected_index = int(self.label2id[class_name])
            else:
                s = f"'{class_name}' is not found in self.label2id keys."
                s += "Defaulting to predicted index instead."
                warnings.warn(s)
                self.selected_index = int(self.predicted_class_index)
        else:
            self.selected_index = int(self.predicted_class_index)

        reference_tokens = [t.replace("Ġ", "") for t in self.decode(self.input_ids)]
        if (self.attribution_type == "lig"):
            explainer = LIGAttributions(
                custom_forward=self._forward,
                embeddings=embeddings,
                tokens=reference_tokens,
                input_ids=self.input_ids,
                ref_input_ids=self.ref_input_ids,
                sep_id=self.sep_idx,
                attention_mask=self.attention_mask,
                position_ids=self.position_ids,
                ref_position_ids=self.ref_position_ids,
                token_type_ids=self.token_type_ids,
                ref_token_type_ids=self.ref_token_type_ids,
                internal_batch_size=self.internal_batch_size,
                n_steps=self.n_steps,
            )
        elif(self.attribution_type == "gs"):
            explainer = GSAttributions(
                custom_forward=self._forward_embs,
                embeddings=self.word_embeddings,
                tokens=reference_tokens,
                input_ids=self.input_ids,
                ref_input_ids=self.ref_input_ids,
                sep_id=self.sep_idx,
                attention_mask=self.attention_mask,
                position_ids=self.position_ids,
                ref_position_ids=self.ref_position_ids,
                token_type_ids=self.token_type_ids,
                ref_token_type_ids=self.ref_token_type_ids,
                target=self.selected_index,
                internal_batch_size=self.internal_batch_size,
                n_samples=self.n_steps,
            )
        else:
            explainer = IGAttributions(
                custom_forward=self._forward_embs,
                embeddings=self.word_embeddings,
                tokens=reference_tokens,
                input_ids=self.input_ids,
                ref_input_ids=self.ref_input_ids,
                sep_id=self.sep_idx,
                attention_mask=self.attention_mask,
                position_ids=self.position_ids,
                ref_position_ids=self.ref_position_ids,
                token_type_ids=self.token_type_ids,
                ref_token_type_ids=self.ref_token_type_ids,
                target=self.selected_index,
                internal_batch_size=self.internal_batch_size,
                n_steps=self.n_steps,
            )
        explainer.summarize(end_idx=self.input_ids.size(1))
        self.attributions = explainer

    def _run(
            self,
            text: str,
            index: int = None,
            class_name: str = None,
            embedding_type: int = None,
    ) -> list:
        if embedding_type is None:
            embeddings = self.word_embeddings
        else:
            if embedding_type == 0:
                embeddings = self.word_embeddings
            elif embedding_type == 1:
                if self.accepts_position_ids and self.position_embeddings is not None:
                    embeddings = self.position_embeddings
                else:
                    warnings.warn(
                        "This model doesn't support position embeddings for attributions. Defaulting to word embeddings"
                    )
                    embeddings = self.word_embeddings
            else:
                embeddings = self.word_embeddings

        self.text = self._clean_text(text)

        self._calculate_attributions(embeddings=embeddings, index=index, class_name=class_name)
        return self.word_attributions

    def __call__(
            self,
            text: str,
            index: int = None,
            class_name: str = None,
            n_steps: int = None,
            internal_batch_size: int = None
    ) -> list:
        if n_steps:
            self.n_steps = n_steps
        if internal_batch_size:
            self.internal_batch_size = internal_batch_size
        return self._run(text, index, class_name)

    def __str__(self):
        s = f"{self.__class__.__name__}("
        s += f"\n\tmodel={self.model.__class__.__name__},"
        s += f"\n\ttokenizer={self.tokenizer.__class__.__name__},"
        s += f"\n\tattribution_type='{self.attribution_type}',"
        s += ")"

        return s


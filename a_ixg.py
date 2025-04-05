"""from typing import Callable, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from captum.attr import InputXGradient
from captum.attr import visualization as viz

from transformers_interpret.errors import AttributionsNotCalculatedError


class Attributions:
    def __init__(self, custom_forward: Callable, embeddings: nn.Module, tokens: list):
        self.custom_forward = custom_forward
        self.embeddings = embeddings
        self.tokens = tokens


class IXGAttributions(Attributions):
    def __init__(
        self,
        custom_forward: Callable,
        embeddings: torch.Tensor,  # tensor, non modulo!
        tokens: list,
        attention_mask: torch.Tensor,
        target: Optional[Union[int, Tuple, torch.Tensor, List]] = None,
    ):
        # Passa il modulo embeddings per coerenza, ma salva il tensore separato
        super().__init__(custom_forward, embeddings, tokens)
        self.target = target
        self.attention_mask = attention_mask

        self.input_x_gradient = InputXGradient(self.custom_forward)

        # Assicurati che gli embeddings richiedano i gradienti
        # Costruisci i parametri da passare a .attribute()
        attr_args = {
            "inputs": embeddings,
            "additional_forward_args": (self.attention_mask,)
        }
        if target is not None:
            attr_args["target"] = target
        self._attributions = self.input_x_gradient.attribute(**attr_args)

    def summarize(self, end_idx=None, flip_sign: bool = False):
        # Somma direttamente le attribuzioni sugli embedding
        self.attributions_sum = self._attributions.sum(dim=-1).squeeze(0)
        if flip_sign:
            self.attributions_sum *= -1
        # Normalizzazione ottimale per IXG:
        # divisione per il valore massimo assoluto (così tutte le attribuzioni sono tra -1 e 1)
        max_abs_val = torch.max(torch.abs(self.attributions_sum[:end_idx]))
        if max_abs_val > 0:
            self.attributions_sum = self.attributions_sum[:end_idx] / max_abs_val

    @property
    def word_attributions(self) -> list:
        if len(self.attributions_sum) >= 1:
            return [(word, float(attr.cpu().data.numpy()))
                    for word, attr in zip(self.tokens, self.attributions_sum)]
        else:
            raise AttributionsNotCalculatedError("Attributions are not yet calculated")

    def visualize_attributions(self, pred_prob, pred_class, true_class, attr_class, all_tokens):
        return viz.VisualizationDataRecord(
            self.attributions_sum,
            pred_prob,
            pred_class,
            true_class,
            attr_class,
            self.attributions_sum.sum(),
            all_tokens,
            None,
        )
"""
from typing import Callable, Optional, List
import torch
from captum.attr import InputXGradient
from captum.attr import visualization as viz
from transformers_interpret.errors import AttributionsNotCalculatedError


class Attributions:
    def __init__(self, custom_forward: Callable, embeddings: torch.Tensor, tokens: List[str]):
        self.custom_forward = custom_forward
        self.embeddings = embeddings
        self.tokens = tokens

class InputXGradientAttributions(Attributions):
    def __init__(
        self,
        custom_forward: Callable,
        embeddings: torch.Tensor,
        tokens: list,
        attention_mask: torch.Tensor,
        target: Optional[int] = None,
    ):
        super().__init__(custom_forward, embeddings, tokens)
        self.attention_mask = attention_mask
        self.target = target

        # Normalizzazione embeddings prima di IXG
        #embeddings = embeddings / torch.norm(embeddings, dim=-1, keepdim=True)

        self.input_x_gradient = InputXGradient(self.custom_forward)
        self._attributions = self.input_x_gradient.attribute(
            inputs=embeddings,
            additional_forward_args=(attention_mask,),
            target=target,
        )

        self.attributions_sum = torch.zeros(len(tokens))

    def summarize(self, end_idx=None, flip_sign: bool = False):
        if flip_sign:
            multiplier = -1
        else:
            multiplier = 1
        self.attributions_sum = self._attributions.sum(dim=-1).squeeze(0) * multiplier
        self.attributions_sum = self.attributions_sum[:end_idx] / torch.norm(self.attributions_sum[:end_idx])

    @property
    def word_attributions(self) -> list:
        wa = []
        if len(self.attributions_sum) >= 1:
            for i, (word, attribution) in enumerate(zip(self.tokens, self.attributions_sum)):
                wa.append((word, float(attribution.cpu().data.numpy())))
            return wa

        else:
            raise AttributionsNotCalculatedError("Attributions are not yet calculated")

    def visualize_attributions(self, pred_prob, pred_class, true_class, attr_class, all_tokens):
        score_viz = viz.VisualizationDataRecord(
            self.attributions_sum,
            pred_prob,
            pred_class,
            true_class,
            attr_class,
            self.attributions_sum.sum(),
            all_tokens,
            delta=None
        )
        return score_viz

    def get_top_k(self, k=5):
        attributions_abs = torch.abs(self.attributions_sum)
        top_k_idx = torch.topk(attributions_abs, k).indices
        return [(self.tokens[i], self.attributions_sum[i].item()) for i in top_k_idx]

    def plot_top_k(self, k=5):
        import matplotlib.pyplot as plt

        top_k = self.get_top_k(k)
        words, scores = zip(*top_k)
        plt.barh(words[::-1], scores[::-1])
        plt.xlabel("Attribution score")
        plt.title(f"Top-{k} tokens by attribution")
        plt.show()

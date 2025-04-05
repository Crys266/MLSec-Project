
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

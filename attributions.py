import torch
import torch.nn as nn
from typing import Callable, List, Optional, Tuple, Union
from captum.attr import GradientShap, IntegratedGradients, LayerIntegratedGradients, visualization as viz
from transformers_interpret.errors import (
    AttributionsNotCalculatedError
)


# Attributions wrapper
class Attributions:
    def __init__(self, custom_forward: Callable, embeddings: nn.Module, tokens: list):
        self.custom_forward = custom_forward
        self.embeddings = embeddings
        self.tokens = tokens


class GSAttributions(Attributions):
    def __init__(
        self,
        custom_forward: Callable,
        embeddings: nn.Module,
        tokens: list,
        input_ids: torch.Tensor,
        ref_input_ids: torch.Tensor,
        sep_id: int,
        attention_mask: torch.Tensor,
        target: Optional[Union[int, Tuple, torch.Tensor, List]] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        ref_token_type_ids: Optional[torch.Tensor] = None,
        ref_position_ids: Optional[torch.Tensor] = None,
        internal_batch_size: Optional[int] = None,
        n_samples: int = 50,
    ):
        super().__init__(custom_forward, embeddings, tokens)
        self.input_ids = input_ids
        self.ref_input_ids = ref_input_ids
        self.attention_mask = attention_mask
        self.target = target
        self.token_type_ids = token_type_ids
        self.position_ids = position_ids
        self.ref_token_type_ids = ref_token_type_ids
        self.ref_position_ids = ref_position_ids
        self.internal_batch_size = internal_batch_size
        self.n_samples = n_samples

        self.gs = GradientShap(self.custom_forward)

        input_embed = self.embeddings(self.input_ids)
        self.baselines = self.embeddings(self.ref_input_ids)
        forward_args = {
            "attention_mask": self.attention_mask
        }
        if self.token_type_ids is not None:
            forward_args["token_type_ids"] = self.token_type_ids
        if self.position_ids is not None:
            forward_args["position_ids"] = self.position_ids

        self._attributions, self.delta = self.gs.attribute(
            inputs=input_embed,
            baselines=self.baselines,
            target=int(self.target),
            additional_forward_args=forward_args,
            return_convergence_delta=True,
            n_samples=self.n_samples,
        )
        # Calcoliamo la media del delta in modo da ottenere uno scalare
        self.delta = torch.mean(self.delta)

    @property
    def word_attributions(self) -> list:
        if len(self.attributions_sum) >= 1:
            return [(word, float(attr.cpu().data.numpy())) for word, attr in zip(self.tokens, self.attributions_sum)]
        else:
            raise AttributionsNotCalculatedError("Attributions are not yet calculated")

    def summarize(self, end_idx=None, flip_sign: bool = False):
        multiplier = -1 if flip_sign else 1
        self.attributions_sum = self._attributions.sum(dim=-1).squeeze(0) * multiplier
        if end_idx is not None:
            self.attributions_sum = self.attributions_sum[:end_idx]
        self.attributions_sum = self.attributions_sum / torch.norm(self.attributions_sum)

    def visualize_attributions(self, pred_prob, pred_class, true_class, attr_class, all_tokens):
        return viz.VisualizationDataRecord(
            self.attributions_sum,
            pred_prob,
            pred_class,
            true_class,
            attr_class,
            self.attributions_sum.sum(),
            all_tokens,
            self.delta,
        )

    @property
    def sensitivity(self):
        """
        Calcola una stima della sensitivity degli attribution aggiungendo un piccolo rumore (eps) agli embeddings.
        Viene calcolata la norma della differenza tra gli attribution originali (sommati su dim -1)
        e quelli ottenuti con un input disturbato, normalizzata rispetto alla norma dell'attribution originale.
        """
        eps = 1e-3
        input_embed = self.embeddings(self.input_ids)
        ref_embed = self.embeddings(self.ref_input_ids)
        forward_args = {"attention_mask": self.attention_mask}
        if self.token_type_ids is not None:
            forward_args["token_type_ids"] = self.token_type_ids
        if self.position_ids is not None:
            forward_args["position_ids"] = self.position_ids

        noise = eps * torch.randn_like(input_embed)
        noisy_input_embed = input_embed + noise

        # Chiamata a gs.attribute senza aspettarsi 2 valori (return_convergence_delta=False)
        noisy_attributions = self.gs.attribute(
            inputs=noisy_input_embed,
            baselines=ref_embed,
            target=int(self.target),
            additional_forward_args=forward_args,
            n_samples=self.n_samples,
            return_convergence_delta=False,
        )
        noisy_attr_sum = noisy_attributions.sum(dim=-1).squeeze(0)
        original_attr_sum = self._attributions.sum(dim=-1).squeeze(0)
        sensitivity = torch.norm(original_attr_sum - noisy_attr_sum) / torch.norm(original_attr_sum)
        return sensitivity


class IGAttributions(Attributions):
    def __init__(
        self,
        custom_forward: Callable,
        embeddings: nn.Module,
        tokens: list,
        input_ids: torch.Tensor,
        ref_input_ids: torch.Tensor,
        sep_id: int,
        attention_mask: torch.Tensor,
        target: Optional[Union[int, Tuple, torch.Tensor, List]] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        ref_token_type_ids: Optional[torch.Tensor] = None,
        ref_position_ids: Optional[torch.Tensor] = None,
        internal_batch_size: Optional[int] = None,
        n_steps: int = 50,
    ):
        super().__init__(custom_forward, embeddings, tokens)
        self.input_ids = input_ids
        self.ref_input_ids = ref_input_ids
        self.attention_mask = attention_mask
        self.target = target
        self.token_type_ids = token_type_ids
        self.position_ids = position_ids
        self.ref_token_type_ids = ref_token_type_ids
        self.ref_position_ids = ref_position_ids
        self.internal_batch_size = internal_batch_size
        self.n_steps = n_steps

        self.ig = IntegratedGradients(self.custom_forward)

        input_embed = self.embeddings(self.input_ids)
        self.baselines = self.embeddings(self.ref_input_ids)
        forward_args = {
            "attention_mask": self.attention_mask
        }
        if self.token_type_ids is not None:
            forward_args["token_type_ids"] = self.token_type_ids
        if self.position_ids is not None:
            forward_args["position_ids"] = self.position_ids

        self._attributions, self.delta = self.ig.attribute(
            inputs=input_embed,
            baselines=self.baselines,
            target=int(self.target),
            additional_forward_args=forward_args,
            return_convergence_delta=True,
            internal_batch_size=self.internal_batch_size,
            n_steps=self.n_steps,
        )

    @property
    def word_attributions(self) -> list:
        wa = []
        if len(self.attributions_sum) >= 1:
            for i, (word, attribution) in enumerate(zip(self.tokens, self.attributions_sum)):
                wa.append((word, float(attribution.cpu().data.numpy())))
            return wa

        else:
            raise AttributionsNotCalculatedError("Attributions are not yet calculated")

    def summarize(self, end_idx=None, flip_sign: bool = False):
        if flip_sign:
            multiplier = -1
        else:
            multiplier = 1
        self.attributions_sum = self._attributions.sum(dim=-1).squeeze(0) * multiplier
        self.attributions_sum = self.attributions_sum[:end_idx] / torch.norm(self.attributions_sum[:end_idx])

    def visualize_attributions(self, pred_prob, pred_class, true_class, attr_class, all_tokens):
        return viz.VisualizationDataRecord(
            self.attributions_sum,
            pred_prob,
            pred_class,
            true_class,
            attr_class,
            self.attributions_sum.sum(),
            all_tokens,
            self.delta,
        )

    @property
    def sensitivity(self):
        """
        Calcola la sensitivity per IG aggiungendo un piccolo rumore agli embeddings.
        Viene valutata la variazione delle attribution (sommate) a fronte della perturbazione.
        """
        eps = 1e-3
        input_embed = self.embeddings(self.input_ids)
        ref_embed = self.embeddings(self.ref_input_ids)
        forward_args = {"attention_mask": self.attention_mask}
        if self.token_type_ids is not None:
            forward_args["token_type_ids"] = self.token_type_ids
        if self.position_ids is not None:
            forward_args["position_ids"] = self.position_ids

        noise = eps * torch.randn_like(input_embed)
        noisy_input_embed = input_embed + noise

        # Chiamata senza ritorno del convergence delta
        noisy_attributions = self.ig.attribute(
            inputs=noisy_input_embed,
            baselines=ref_embed,
            target=int(self.target),
            additional_forward_args=forward_args,
            internal_batch_size=self.internal_batch_size,
            n_steps=self.n_steps,
            return_convergence_delta=False,
        )
        noisy_attr_sum = noisy_attributions.sum(dim=-1).squeeze(0)
        original_attr_sum = self._attributions.sum(dim=-1).squeeze(0)
        sensitivity = torch.norm(original_attr_sum - noisy_attr_sum) / torch.norm(original_attr_sum)
        return sensitivity



class LIGAttributions(Attributions):
    def __init__(
        self,
        custom_forward: Callable,
        embeddings: nn.Module,
        tokens: list,
        input_ids: torch.Tensor,
        ref_input_ids: torch.Tensor,
        sep_id: int,
        attention_mask: torch.Tensor,
        target: Optional[Union[int, Tuple, torch.Tensor, List]] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        ref_token_type_ids: Optional[torch.Tensor] = None,
        ref_position_ids: Optional[torch.Tensor] = None,
        internal_batch_size: Optional[int] = None,
        n_steps: int = 50,
    ):
        super().__init__(custom_forward, embeddings, tokens)
        self.input_ids = input_ids
        self.ref_input_ids = ref_input_ids
        self.attention_mask = attention_mask
        self.target = target
        self.token_type_ids = token_type_ids
        self.position_ids = position_ids
        self.ref_token_type_ids = ref_token_type_ids
        self.ref_position_ids = ref_position_ids
        self.internal_batch_size = internal_batch_size
        self.n_steps = n_steps

        self.lig = LayerIntegratedGradients(self.custom_forward, self.embeddings)
        self.baselines = (
            self.ref_input_ids,
            self.ref_token_type_ids,
            self.ref_position_ids,
        )

        if self.token_type_ids is not None and self.position_ids is not None:
            self._attributions, self.delta = self.lig.attribute(
                inputs=(self.input_ids, self.token_type_ids, self.position_ids),
                baselines=self.baselines,
                target=self.target,
                return_convergence_delta=True,
                additional_forward_args=(self.attention_mask),
                internal_batch_size=self.internal_batch_size,
                n_steps=self.n_steps,
            )
        elif self.position_ids is not None:
            self._attributions, self.delta = self.lig.attribute(
                inputs=(self.input_ids, self.position_ids),
                baselines=(
                    self.ref_input_ids,
                    self.ref_position_ids,
                ),
                target=self.target,
                return_convergence_delta=True,
                additional_forward_args=(self.attention_mask),
                internal_batch_size=self.internal_batch_size,
                n_steps=self.n_steps,
            )
        elif self.token_type_ids is not None:
            self._attributions, self.delta = self.lig.attribute(
                inputs=(self.input_ids, self.token_type_ids),
                baselines=(
                    self.ref_input_ids,
                    self.ref_token_type_ids,
                ),
                target=self.target,
                return_convergence_delta=True,
                additional_forward_args=(self.attention_mask),
                internal_batch_size=self.internal_batch_size,
                n_steps=self.n_steps,
            )

        else:
            self._attributions, self.delta = self.lig.attribute(
                inputs=self.input_ids,
                baselines=self.ref_input_ids,
                target=self.target,
                return_convergence_delta=True,
                internal_batch_size=self.internal_batch_size,
                n_steps=self.n_steps,
            )

    @property
    def word_attributions(self) -> list:
        wa = []
        if len(self.attributions_sum) >= 1:
            for i, (word, attribution) in enumerate(zip(self.tokens, self.attributions_sum)):
                wa.append((word, float(attribution.cpu().data.numpy())))
            return wa

        else:
            raise AttributionsNotCalculatedError("Attributions are not yet calculated")

    def summarize(self, end_idx=None, flip_sign: bool = False):
        if flip_sign:
            multiplier = -1
        else:
            multiplier = 1
        self.attributions_sum = self._attributions.sum(dim=-1).squeeze(0) * multiplier
        self.attributions_sum = self.attributions_sum[:end_idx] / torch.norm(self.attributions_sum[:end_idx])

    def visualize_attributions(self, pred_prob, pred_class, true_class, attr_class, all_tokens):

        return viz.VisualizationDataRecord(
            self.attributions_sum,
            pred_prob,
            pred_class,
            true_class,
            attr_class,
            self.attributions_sum.sum(),
            all_tokens,
            self.delta,
        )

    @property
    def sensitivity(self):
        """
        Computes sensitivity for LIG by perturbing the input embeddings.
        Since LIG normally works on discrete inputs, we bypass this by computing
        on the embedding space using a temporary forward function.
        """
        eps = 1e-3
        orig_embed = self.embeddings(self.input_ids)
        baseline_embed = self.embeddings(self.ref_input_ids)
        noisy_embed = orig_embed + eps * torch.randn_like(orig_embed)

        # Retrieve the model from the bound custom_forward method.
        model = self.custom_forward.__self__.model

        def custom_forward_for_sens(embeds):
            logits = model(
                inputs_embeds=embeds,
                attention_mask=self.attention_mask,
                token_type_ids=self.token_type_ids,
                position_ids=self.position_ids
            ).logits
            # Select the target logit; if self.target is not provided, default to 0.
            tgt = self.target if self.target is not None else 0
            return logits[:, tgt]

        lig_sens = IntegratedGradients(custom_forward_for_sens)
        # Now, since custom_forward_for_sens returns a scalar per sample,
        # we do not pass the target to attribute.
        orig_attr = lig_sens.attribute(
            inputs=orig_embed,
            baselines=baseline_embed,
            n_steps=self.n_steps,
            return_convergence_delta=False,
        )
        noisy_attr = lig_sens.attribute(
            inputs=noisy_embed,
            baselines=baseline_embed,
            n_steps=self.n_steps,
            return_convergence_delta=False,
        )
        orig_attr_sum = orig_attr.sum(dim=-1).squeeze(0)
        noisy_attr_sum = noisy_attr.sum(dim=-1).squeeze(0)
        sensitivity = torch.norm(orig_attr_sum - noisy_attr_sum) / torch.norm(orig_attr_sum)
        return sensitivity

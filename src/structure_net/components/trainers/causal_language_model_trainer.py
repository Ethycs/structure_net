"""Canonical trainer component for Structure Net causal language models."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple, Type

import torch

from ...core.base_components import BaseTrainer
from ...core.interfaces import (
    ComponentContract,
    ComponentVersion,
    EvolutionContext,
    IModel,
    Maturity,
    ResourceLevel,
    ResourceRequirements,
)


class CausalLanguageModelTrainer(BaseTrainer):
    """Train tuple-returning causal LMs and remain compatible with growth.

    Tensor-only batches are shifted into next-token input/target pairs. Mapping
    and tuple batches may supply already aligned ``input_ids`` and ``targets``.
    The optimizer is public so an evolver can register parameters introduced
    after training has begun.
    """

    def __init__(
        self,
        *,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.AdamW,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        max_grad_norm: Optional[float] = 1.0,
        refinement_steps: Optional[int] = None,
        name: Optional[str] = None,
    ):
        super().__init__(name or "CausalLanguageModelTrainer")
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs or {"lr": 3e-4, "weight_decay": 0.0}
        self.max_grad_norm = max_grad_norm
        self.refinement_steps = refinement_steps
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.model_identity: Optional[int] = None
        self.step_count = 0
        self._contract = ComponentContract(
            component_name=self.name,
            version=ComponentVersion(1, 0, 0),
            maturity=Maturity.EXPERIMENTAL,
            required_inputs={"model", "batch.input_ids"},
            optional_inputs={"batch.targets", "model.refinement_steps"},
            provided_outputs={"training.loss", "training.grad_norm", "training.tokens"},
            resources=ResourceRequirements(
                memory_level=ResourceLevel.HIGH,
                requires_gpu=False,
                parallel_safe=False,
            ),
        )

    @property
    def contract(self) -> ComponentContract:
        return self._contract

    def attach(self, model: IModel) -> torch.optim.Optimizer:
        """Create or return the optimizer attached to ``model``."""
        if self.optimizer is not None and self.model_identity != id(model):
            raise ValueError("Trainer is already attached to a different model")
        if self.optimizer is None:
            self.optimizer = self.optimizer_class(
                model.parameters(), **self.optimizer_kwargs
            )
            self.model_identity = id(model)
        return self.optimizer

    @staticmethod
    def _prepare_batch(batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(batch, Mapping):
            input_ids = batch.get("input_ids")
            targets = batch.get("targets", batch.get("labels"))
            if input_ids is None:
                raise ValueError("Mapping batch must contain input_ids")
            if targets is None:
                if input_ids.shape[1] < 2:
                    raise ValueError("An unlabelled token batch needs at least two positions")
                return input_ids[:, :-1], input_ids[:, 1:]
            return input_ids, targets
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            return batch[0], batch[1]
        if isinstance(batch, torch.Tensor):
            if batch.ndim != 2 or batch.shape[1] < 2:
                raise ValueError("A token batch must have shape [batch, sequence>=2]")
            return batch[:, :-1], batch[:, 1:]
        raise TypeError("Batch must be a token tensor, mapping, or (input_ids, targets)")

    def _train_step(
        self, model: IModel, batch: Any, context: EvolutionContext
    ) -> Dict[str, float]:
        optimizer = self.attach(model)
        device = torch.device(context.get("device", next(model.parameters()).device))
        input_ids, targets = self._prepare_batch(batch)
        input_ids = input_ids.to(device)
        targets = targets.to(device)

        model.train()
        optimizer.zero_grad(set_to_none=True)
        output = model(
            input_ids,
            targets,
            refinement_steps=context.get("refinement_steps", self.refinement_steps),
            return_full_logits=True,
        )
        if not isinstance(output, tuple) or len(output) != 2 or output[1] is None:
            raise TypeError("Causal LM must return (logits, loss) when targets are supplied")
        loss = output[1]
        loss.backward()
        if self.max_grad_norm is None:
            grad_norm = torch.linalg.vector_norm(
                torch.stack(
                    [
                        parameter.grad.detach().norm()
                        for parameter in model.parameters()
                        if parameter.grad is not None
                    ]
                )
            )
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), self.max_grad_norm
            )
        optimizer.step()
        self.step_count += 1
        return {
            "loss": float(loss.detach().cpu()),
            "grad_norm": float(grad_norm.detach().cpu()),
            "tokens": float(targets.numel()),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "step": float(self.step_count),
            "refinement_steps": float(
                context.get(
                    "refinement_steps",
                    self.refinement_steps
                    if self.refinement_steps is not None
                    else getattr(model, "refinement_steps", 0),
                )
            ),
        }


__all__ = ["CausalLanguageModelTrainer"]

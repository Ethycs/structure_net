"""Two-stage trainer for CALM-style continuous autoregressive models."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Type

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


class ContinuousAutoregressiveTrainer(BaseTrainer):
    """Train either the chunk autoencoder or the continuous language model.

    A trainer instance owns one phase so that optimizer state and gradient
    routing cannot silently cross the two-stage training boundary.
    """

    VALID_PHASES = {"autoencoder", "energy"}

    def __init__(
        self,
        *,
        phase: str,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.AdamW,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        max_grad_norm: Optional[float] = 1.0,
        name: Optional[str] = None,
    ):
        if phase not in self.VALID_PHASES:
            raise ValueError(f"phase must be one of {sorted(self.VALID_PHASES)}")
        super().__init__(name or f"ContinuousAutoregressiveTrainer_{phase}")
        self.phase = phase
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs or {"lr": 3e-4, "weight_decay": 0.0}
        self.max_grad_norm = max_grad_norm
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.model_identity: Optional[int] = None
        self.step_count = 0
        self._optimized_parameters: list[torch.nn.Parameter] = []
        self._contract = ComponentContract(
            component_name=self.name,
            version=ComponentVersion(1, 0, 0),
            maturity=Maturity.EXPERIMENTAL,
            required_inputs={"model", "batch.input_ids"},
            provided_outputs={
                "training.loss",
                "training.grad_norm",
                "training.tokens",
                "training.phase",
            },
            resources=ResourceRequirements(
                memory_level=ResourceLevel.HIGH,
                requires_gpu=False,
                parallel_safe=False,
            ),
        )

    @property
    def contract(self) -> ComponentContract:
        return self._contract

    def supports_online_evolution(self) -> bool:
        return False

    def attach(self, model: IModel) -> torch.optim.Optimizer:
        if self.optimizer is not None and self.model_identity != id(model):
            raise ValueError("trainer is already attached to a different model")
        if self.optimizer is not None:
            return self.optimizer
        if not hasattr(model, "autoencoder") or not hasattr(model, "set_autoencoder_frozen"):
            raise TypeError("continuous autoregressive training requires a CALM-style model")

        if self.phase == "autoencoder":
            model.set_autoencoder_frozen(False)
            self._optimized_parameters = list(model.autoencoder.parameters())
        else:
            model.set_autoencoder_frozen(True)
            self._optimized_parameters = [
                parameter for parameter in model.parameters() if parameter.requires_grad
            ]
        self.optimizer = self.optimizer_class(
            self._optimized_parameters, **self.optimizer_kwargs
        )
        self.model_identity = id(model)
        return self.optimizer

    @staticmethod
    def _prepare_batch(batch: Any) -> torch.Tensor:
        if isinstance(batch, Mapping):
            input_ids = batch.get("input_ids")
            if input_ids is None:
                raise ValueError("mapping batch must contain input_ids")
            return input_ids
        if isinstance(batch, torch.Tensor):
            return batch
        raise TypeError("batch must be a token tensor or mapping with input_ids")

    def _train_step(
        self, model: IModel, batch: Any, context: EvolutionContext
    ) -> Dict[str, float]:
        optimizer = self.attach(model)
        device = torch.device(context.get("device", next(model.parameters()).device))
        input_ids = self._prepare_batch(batch).to(device)
        model.train()
        if self.phase == "autoencoder":
            model.autoencoder.train()

        optimizer.zero_grad(set_to_none=True)
        if self.phase == "autoencoder":
            output = model.autoencoder(input_ids)
            loss = output.loss
            extras = {
                "reconstruction_loss": float(output.reconstruction_loss.detach().cpu()),
                "kl_loss": float(output.kl_loss.detach().cpu()),
            }
        else:
            output = model(input_ids)
            loss = output.loss
            extras = {"energy_loss": float(loss.detach().cpu())}
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite {self.phase} loss")
        loss.backward()

        gradients = [
            parameter.grad.detach().norm()
            for parameter in self._optimized_parameters
            if parameter.grad is not None
        ]
        if not gradients:
            raise RuntimeError(f"{self.phase} phase produced no gradients")
        if self.max_grad_norm is None:
            grad_norm = torch.linalg.vector_norm(torch.stack(gradients))
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self._optimized_parameters, self.max_grad_norm
            )
        if not torch.isfinite(grad_norm):
            raise FloatingPointError(f"non-finite {self.phase} gradient norm")
        optimizer.step()
        self.step_count += 1

        return {
            "loss": float(loss.detach().cpu()),
            "grad_norm": float(grad_norm.detach().cpu()),
            "tokens": float(input_ids.numel()),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "step": float(self.step_count),
            "phase_autoencoder": float(self.phase == "autoencoder"),
            **extras,
        }


__all__ = ["ContinuousAutoregressiveTrainer"]

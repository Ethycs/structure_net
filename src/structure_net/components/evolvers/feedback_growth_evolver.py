"""Evolver for adding delayed backward-feedback patches to language models."""

from __future__ import annotations

from typing import Any, Dict

from ...core import (
    BaseEvolver,
    ComponentContract,
    ComponentVersion,
    EvolutionPlan,
    IModel,
    ITrainer,
    Maturity,
    ResourceLevel,
    ResourceRequirements,
)


class FeedbackGrowthEvolver(BaseEvolver):
    """Apply explicit or random feedback-growth plans to compatible models."""

    PLAN_TYPES = {"add_feedback_connection", "add_random_feedback_connections"}

    def __init__(self, name: str | None = None):
        super().__init__(name or "FeedbackGrowthEvolver")
        self._supported_plan_types = self.PLAN_TYPES.copy()
        self._contract = ComponentContract(
            component_name=self.name,
            version=ComponentVersion(1, 0, 0),
            maturity=Maturity.EXPERIMENTAL,
            required_inputs={"model.dynamic_feedback", "plans.feedback_growth"},
            provided_outputs={"evolution.feedback_connections"},
            resources=ResourceRequirements(
                memory_level=ResourceLevel.MEDIUM,
                requires_gpu=False,
                parallel_safe=False,
            ),
        )

    @property
    def contract(self) -> ComponentContract:
        return self._contract

    def can_execute_plan(self, plan: EvolutionPlan) -> bool:
        return plan.get("type") in self.PLAN_TYPES

    @staticmethod
    def _register_new_parameters(optimizer: Any, parameters: list[Any]) -> int:
        if optimizer is None:
            return 0
        known = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        new_parameters = [parameter for parameter in parameters if id(parameter) not in known]
        if new_parameters:
            optimizer.add_param_group({"params": new_parameters})
        return sum(parameter.numel() for parameter in new_parameters)

    def _execute_plan(
        self,
        plan: EvolutionPlan,
        model: IModel,
        trainer: ITrainer,
        optimizer: Any,
    ) -> Dict[str, Any]:
        plan_type = plan["type"]
        common = {
            "num_neurons": int(plan.get("num_neurons", 16)),
            "connection_density": float(plan.get("connection_density", 1.0)),
            "gate_init": float(plan.get("gate_init", 0.0)),
            "seed": int(plan.get("seed", 0)),
        }

        if plan_type == "add_feedback_connection":
            add = getattr(model, "add_feedback_connection", None)
            if add is None:
                raise TypeError("Model does not support explicit feedback growth")
            patches = [
                add(
                    source_layer=int(plan["source_layer"]),
                    target_layer=int(plan["target_layer"]),
                    **common,
                )
            ]
        else:
            add_random = getattr(model, "add_random_feedback_connections", None)
            if add_random is None:
                raise TypeError("Model does not support random feedback growth")
            patches = add_random(count=int(plan.get("count", 1)), **common)

        if "refinement_steps" in plan:
            refinement_steps = int(plan["refinement_steps"])
            if refinement_steps < 1:
                raise ValueError("A feedback model requires at least one refinement step")
            model.refinement_steps = refinement_steps

        registered_parameters = self._register_new_parameters(
            optimizer,
            [parameter for patch in patches for parameter in patch.parameters()],
        )
        return {
            "success": True,
            "action_taken": plan_type,
            "connections_added": len(patches),
            "optimizer_parameters_added": registered_parameters,
            "refinement_steps": model.refinement_steps,
            "feedback_connections": [
                patch.connectivity_summary() for patch in patches
            ],
        }


__all__ = ["FeedbackGrowthEvolver"]

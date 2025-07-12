from src.structure_net.core.base_components import BaseOrchestrator
from src.structure_net.core.interfaces import (
    IScheduler,
    EvolutionContext,
    ComponentContract,
    ComponentVersion,
    Maturity,
    ResourceRequirements,
    ResourceLevel,
)
from typing import List
import logging

class AdaptiveLearningRateOrchestrator(BaseOrchestrator):
    """Orchestrates learning rate adjustments from multiple schedulers."""

    def __init__(self, schedulers: List[IScheduler], name: str = None):
        super().__init__(name or "AdaptiveLearningRateOrchestrator")
        self.schedulers = schedulers

    @property
    def contract(self) -> ComponentContract:
        """Declares the contract for this component."""
        return ComponentContract(
            component_name=self.name,
            version=ComponentVersion(1, 0, 0),
            maturity=Maturity.STABLE,
            resources=ResourceRequirements(
                memory_level=ResourceLevel.LOW,
                requires_gpu=False
            ),
        )

    def run_cycle(self, context: EvolutionContext) -> EvolutionContext:
        """Adjusts learning rates based on all managed schedulers."""
        self.log(logging.INFO, "Starting learning rate adjustment cycle.")
        
        # This is a simplified approach. A real implementation would need to handle
        # optimizer parameter groups and potentially conflicting suggestions.
        final_lrs = {}

        for scheduler in self.schedulers:
            try:
                if scheduler.should_trigger(context):
                    lr_update = scheduler.propose_plan(None, context) # Schedulers might not need a full report
                    if lr_update:
                        # In a real system, you'd merge these updates intelligently
                        final_lrs.update(lr_update.get("learning_rate_adjustments", {}))
            except Exception as e:
                self.log(logging.ERROR, f"Scheduler {scheduler.name} failed: {e}")
        
        if final_lrs:
            # Apply the final learning rates to the optimizer in the context
            optimizer = context.get("optimizer")
            if optimizer:
                for param_group in optimizer.param_groups:
                    group_name = param_group.get("name", "default")
                    if group_name in final_lrs:
                        param_group['lr'] = final_lrs[group_name].get("new_lr", param_group['lr'])
                self.log(logging.INFO, f"Updated learning rates for {len(final_lrs)} groups.")
            else:
                self.log(logging.WARNING, "No optimizer found in context to apply learning rates.")

        self.log(logging.INFO, "Learning rate adjustment cycle complete.")
        return context

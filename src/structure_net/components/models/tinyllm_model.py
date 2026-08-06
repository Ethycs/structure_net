"""TinyLLM-compatible GPT-2 model with optional recurrent feedback growth.

The baseline graph mirrors the GPT-2 implementation pinned by
``weiserlab/TinyLLM``.  Experimental backward skips are evaluated on a later
refinement pass so the graph always has an explicit execution order.
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...core import (
    BaseLayer,
    BaseModel,
    ComponentContract,
    ComponentVersion,
    ILayer,
    Maturity,
    ResourceLevel,
    ResourceRequirements,
)


TINYLLM_PRESETS: Dict[str, Tuple[int, int, int]] = {
    "d6": (6, 6, 384),
    "d8": (8, 8, 512),
    "d10": (10, 10, 640),
    "d11": (11, 11, 704),
    "d12": (12, 12, 768),
}

_PRESET_ALIASES = {
    "30m": "d6",
    "51m": "d8",
    "82m": "d10",
    "101m": "d11",
    "102m": "d11",
    "124m": "d12",
    "gpt2": "d12",
}


@dataclass(frozen=True)
class TinyLLMConfig:
    """Configuration for the GPT-2 family used by TinyLLM."""

    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    default_refinement_steps: int = 0
    capture_activations: bool = False
    initialization_seed: int = 42

    def __post_init__(self) -> None:
        integer_fields = {
            "block_size": self.block_size,
            "vocab_size": self.vocab_size,
            "n_layer": self.n_layer,
            "n_head": self.n_head,
            "n_embd": self.n_embd,
        }
        for name, value in integer_fields.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.n_embd % self.n_head:
            raise ValueError("n_embd must be divisible by n_head")
        if self.default_refinement_steps < 0:
            raise ValueError("default_refinement_steps cannot be negative")

    @classmethod
    def from_preset(cls, preset: str = "d6", **overrides: Any) -> "TinyLLMConfig":
        """Build one of TinyLLM's 30M--124M model configurations."""
        normalized = preset.lower().replace("gpt2:", "")
        normalized = _PRESET_ALIASES.get(normalized, normalized)
        if normalized not in TINYLLM_PRESETS:
            available = ", ".join(TINYLLM_PRESETS)
            raise ValueError(f"Unknown TinyLLM preset {preset!r}; choose one of {available}")
        n_layer, n_head, n_embd = TINYLLM_PRESETS[normalized]
        values = {"n_layer": n_layer, "n_head": n_head, "n_embd": n_embd}
        values.update(overrides)
        return cls(**values)

    @property
    def preset(self) -> Optional[str]:
        shape = (self.n_layer, self.n_head, self.n_embd)
        return next((name for name, value in TINYLLM_PRESETS.items() if value == shape), None)

    @classmethod
    def from_huggingface_config(cls, config: Any, **overrides: Any) -> "TinyLLMConfig":
        """Build from a ``transformers.GPT2Config`` object or plain mapping."""

        def read(*names: str) -> int:
            for field_name in names:
                if isinstance(config, Mapping) and field_name in config:
                    return int(config[field_name])
                if hasattr(config, field_name):
                    return int(getattr(config, field_name))
            raise ValueError(f"Missing Hugging Face config field; tried {names}")

        values = {
            "block_size": read("n_positions", "n_ctx", "block_size"),
            "vocab_size": read("vocab_size"),
            "n_layer": read("n_layer", "num_hidden_layers"),
            "n_head": read("n_head", "num_attention_heads"),
            "n_embd": read("n_embd", "hidden_size"),
        }
        values.update(overrides)
        return cls(**values)

    def to_huggingface_config(self) -> Dict[str, Any]:
        """Return the architecture fields consumed by ``GPT2Config``."""
        return {
            "model_type": "gpt2",
            "architectures": ["GPT2LMHeadModel"],
            "vocab_size": self.vocab_size,
            "n_positions": self.block_size,
            "n_ctx": self.block_size,
            "n_embd": self.n_embd,
            "n_layer": self.n_layer,
            "n_head": self.n_head,
        }


@dataclass
class TinyLLMOutput:
    """Optional structured result for analysis-aware callers."""

    logits: torch.Tensor
    loss: Optional[torch.Tensor]
    hidden_states: Optional[Tuple[torch.Tensor, ...]]
    refinement_steps: int


class TinyLLMGELU(nn.Module):
    """The tanh GELU variant used by GPT-2 and TinyLLM's llm.c reference."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        coefficient = math.sqrt(2.0 / math.pi)
        return 0.5 * value * (1.0 + torch.tanh(coefficient * (value + 0.044715 * value.pow(3))))


class TinyLLMAttention(nn.Module):
    def __init__(self, config: TinyLLMConfig):
        super().__init__()
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj._tinyllm_residual_projection = True
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, channels = value.shape
        query, key, content = self.c_attn(value).split(self.n_embd, dim=-1)
        head_size = channels // self.n_head

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(batch_size, sequence_length, self.n_head, head_size).transpose(1, 2)

        query, key, content = map(split_heads, (query, key, content))
        attended = F.scaled_dot_product_attention(
            query,
            key,
            content,
            dropout_p=0.0,
            is_causal=True,
        )
        attended = attended.transpose(1, 2).contiguous().view(batch_size, sequence_length, channels)
        return self.c_proj(attended)


class TinyLLMMLP(nn.Module):
    def __init__(self, config: TinyLLMConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = TinyLLMGELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj._tinyllm_residual_projection = True

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.c_proj(self.gelu(self.c_fc(value)))


class TinyLLMBlock(BaseLayer):
    """A pre-layer-normalized TinyLLM transformer block."""

    def __init__(self, config: TinyLLMConfig, index: int):
        super().__init__(name=f"TinyLLMBlock_{index}")
        self.index = index
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = TinyLLMAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = TinyLLMMLP(config)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = value + self.attn(self.ln_1(value))
        return value + self.mlp(self.ln_2(value))

    def forward_gated(self, value: torch.Tensor, gate: float) -> torch.Tensor:
        """Continuously activate this residual block for ``gate`` in ``[0, 1]``.

        Both residual branches use the same gate, while the MLP observes the
        partially updated attention state. The endpoints are exact: zero is
        the identity and one is the ordinary block forward pass.
        """
        amount = float(gate)
        if not 0.0 <= amount <= 1.0:
            raise ValueError("gate must be in [0, 1]")
        if amount == 0.0:
            return value
        if amount == 1.0:
            return self(value)
        attended = value + amount * self.attn(self.ln_1(value))
        return attended + amount * self.mlp(self.ln_2(attended))

    def get_analysis_properties(self) -> Dict[str, torch.Tensor]:
        return {
            "weights.attention_qkv": self.attn.c_attn.weight.detach(),
            "weights.attention_projection": self.attn.c_proj.weight.detach(),
            "weights.mlp_input": self.mlp.c_fc.weight.detach(),
            "weights.mlp_projection": self.mlp.c_proj.weight.detach(),
        }

    def supports_modification(self) -> bool:
        return False

    def add_connections(self, num_connections: int, **kwargs: Any) -> bool:
        return False


class RecurrentFeedbackConnection(BaseLayer):
    """A low-rank delayed connection from a later block to an earlier block."""

    def __init__(
        self,
        channels: int,
        width: int,
        source_block: int,
        target_block: int,
        *,
        initial_gate: float = 0.0,
        connection_density: float = 1.0,
        seed: int = 0,
        name: Optional[str] = None,
    ):
        super().__init__(name=name or f"Feedback_{source_block}_to_{target_block}")
        if width <= 0:
            raise ValueError("Feedback width must be positive")
        if target_block >= source_block:
            raise ValueError("A backward connection requires target_block < source_block")
        if not 0.0 < connection_density <= 1.0:
            raise ValueError("connection_density must be in (0, 1]")

        self.channels = channels
        self.width = width
        self.source_block = source_block
        self.target_block = target_block
        self.connection_density = connection_density
        self.source_weight = nn.Parameter(torch.empty(width, channels))
        self.target_weight = nn.Parameter(torch.empty(channels, width))
        self.register_buffer(
            "source_mask", torch.zeros_like(self.source_weight, dtype=torch.bool)
        )
        self.register_buffer(
            "target_mask", torch.zeros_like(self.target_weight, dtype=torch.bool)
        )
        self.gate = nn.Parameter(torch.tensor(float(initial_gate)))
        self.activation = TinyLLMGELU()
        self._reset_parameters(seed)

    def _reset_parameters(self, seed: int) -> None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        nn.init.normal_(self.source_weight, mean=0.0, std=0.02, generator=generator)
        nn.init.normal_(
            self.target_weight,
            mean=0.0,
            std=0.02 / math.sqrt(max(1, self.width)),
            generator=generator,
        )
        fan = max(1, round(self.channels * self.connection_density))
        for neuron in range(self.width):
            source_indices = torch.randperm(self.channels, generator=generator)[:fan]
            target_indices = torch.randperm(self.channels, generator=generator)[:fan]
            self.source_mask[neuron, source_indices] = 1
            self.target_mask[target_indices, neuron] = 1

    def forward(self, source_state: torch.Tensor) -> torch.Tensor:
        hidden = F.linear(source_state, self.source_weight * self.source_mask)
        correction = F.linear(
            self.activation(hidden), self.target_weight * self.target_mask
        )
        return torch.tanh(self.gate) * correction

    def get_analysis_properties(self) -> Dict[str, torch.Tensor]:
        return {
            "weights.source_projection": self.source_weight.detach(),
            "weights.target_projection": self.target_weight.detach(),
            "masks.source_projection": self.source_mask.detach(),
            "masks.target_projection": self.target_mask.detach(),
            "feedback.gate": self.gate.detach(),
        }

    def supports_modification(self) -> bool:
        return True

    def add_connections(self, num_connections: int, **kwargs: Any) -> bool:
        if num_connections <= 0:
            return False
        combined = torch.cat(
            [
                (self.source_mask == 0).flatten().nonzero().flatten(),
                (self.target_mask == 0).flatten().nonzero().flatten()
                + self.source_mask.numel(),
            ]
        )
        if combined.numel() == 0:
            return False
        generator = kwargs.get("generator")
        selected = combined[
            torch.randperm(combined.numel(), generator=generator)[:num_connections]
        ]
        source_size = self.source_mask.numel()
        source_indices = selected[selected < source_size]
        target_indices = selected[selected >= source_size] - source_size
        self.source_mask.view(-1)[source_indices] = 1
        self.target_mask.view(-1)[target_indices] = 1
        return True

    @property
    def source_layer(self) -> int:
        return self.source_block

    @property
    def target_layer(self) -> int:
        return self.target_block

    @property
    def num_neurons(self) -> int:
        return self.width

    @property
    def active_connections(self) -> int:
        return int(self.source_mask.sum().item() + self.target_mask.sum().item())

    def describe(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "source_block": self.source_block,
            "target_block": self.target_block,
            "width": self.width,
            "num_neurons": self.width,
            "connection_density": self.connection_density,
            "active_connections": self.active_connections,
            "gate": float(torch.tanh(self.gate.detach()).cpu()),
            "execution": "previous refinement pass",
            "parameters": sum(parameter.numel() for parameter in self.parameters()),
        }

    def connectivity_summary(self) -> Dict[str, Any]:
        return self.describe()


# The public name describes the experimental role; retain the recurrent name
# as a compatibility alias for early adopters of this branch.
BackwardFeedbackPatch = RecurrentFeedbackConnection


class TinyLLMModel(BaseModel):
    """TinyLLM's GPT-2 model exposed through Structure Net contracts."""

    CHECKPOINT_FORMAT = "structure_net.tinyllm_feedback"
    CHECKPOINT_VERSION = 1

    _HF_TRANSPOSED_SUFFIXES = (
        "attn.c_attn.weight",
        "attn.c_proj.weight",
        "mlp.c_fc.weight",
        "mlp.c_proj.weight",
    )

    def __init__(self, config: TinyLLMConfig, name: Optional[str] = None):
        super().__init__(name or "TinyLLMModel")
        self.config = config
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "wpe": nn.Embedding(config.block_size, config.n_embd),
                "h": nn.ModuleList([TinyLLMBlock(config, index) for index in range(config.n_layer)]),
                "ln_f": nn.LayerNorm(config.n_embd),
            }
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head._tinyllm_skip_initialization = True
        self.transformer["wte"].weight = self.lm_head.weight
        self.feedback_connections = nn.ModuleList()
        self.refinement_steps = config.default_refinement_steps
        self.layer_activations: List[torch.Tensor] = []
        self.checkpoint_metadata: Optional[Dict[str, Any]] = None
        self._layers = list(self.transformer["h"])
        self._initialize_baseline()

        self._contract = ComponentContract(
            component_name=self.name,
            version=ComponentVersion(1, 0, 0),
            maturity=Maturity.EXPERIMENTAL,
            required_inputs={"input_ids"},
            optional_inputs={"targets", "refinement_steps"},
            provided_outputs={
                "model.logits",
                "model.loss",
                "model.hidden_states",
                "model.architecture",
                "model.feedback_topology",
                "model.dynamic_feedback",
            },
            resources=ResourceRequirements(
                memory_level=ResourceLevel.HIGH,
                requires_gpu=False,
                parallel_safe=True,
            ),
        )

    @property
    def contract(self) -> ComponentContract:
        return self._contract

    def _initialize_baseline(self) -> None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.config.initialization_seed)

        def initialize(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                if getattr(module, "_tinyllm_skip_initialization", False):
                    return
                standard_deviation = 0.02
                if getattr(module, "_tinyllm_residual_projection", False):
                    standard_deviation /= math.sqrt(2 * self.config.n_layer)
                nn.init.normal_(module.weight, mean=0.0, std=standard_deviation, generator=generator)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02, generator=generator)

        self.apply(initialize)

    def _run_blocks(
        self,
        embedded: torch.Tensor,
        previous_states: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        value = embedded
        states: List[torch.Tensor] = []
        feedback_by_target: Dict[int, List[RecurrentFeedbackConnection]] = {}
        if previous_states is not None:
            for connection in self.feedback_connections:
                feedback_by_target.setdefault(connection.target_block, []).append(connection)

        for index, block in enumerate(self.transformer["h"]):
            for connection in feedback_by_target.get(index, []):
                value = value + connection(previous_states[connection.source_block])
            value = block(value)
            states.append(value)
        return value, states

    def _run_blocks_to_depth(
        self,
        embedded: torch.Tensor,
        depth: float,
    ) -> torch.Tensor:
        """Run an exact prefix plus one continuously gated residual block."""
        requested = float(depth)
        maximum = float(self.config.n_layer)
        if not 0.0 <= requested <= maximum:
            raise ValueError(f"depth must be in [0, {self.config.n_layer}]")
        integer_depth = min(int(math.floor(requested)), self.config.n_layer)
        value = embedded
        for index in range(integer_depth):
            value = self.transformer["h"][index](value)
        fraction = requested - integer_depth
        if integer_depth < self.config.n_layer and fraction > 0.0:
            value = self.transformer["h"][integer_depth].forward_gated(
                value, fraction
            )
        return value

    def forward_at_depth(
        self,
        input_ids: torch.Tensor,
        depth: float,
        *,
        return_full_logits: bool = False,
    ) -> torch.Tensor:
        """Evaluate the shared LM head at a real-valued transformer depth.

        Integer depths reproduce exact block prefixes; an intermediate depth
        partially activates the following attention and MLP residuals. This
        path intentionally excludes recurrent feedback refinements.
        """
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape (batch, sequence)")
        _, sequence_length = input_ids.shape
        if sequence_length > self.config.block_size:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds block size {self.config.block_size}"
            )
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("input_ids must contain integer token ids")
        if self.feedback_connections:
            raise ValueError("continuous depth does not support feedback connections")
        positions = torch.arange(
            sequence_length, device=input_ids.device, dtype=torch.long
        )
        embedded = self.transformer["wte"](input_ids) + self.transformer["wpe"](
            positions
        )
        value = self._run_blocks_to_depth(embedded, depth)
        value = self.transformer["ln_f"](value)
        logits_input = value if return_full_logits else value[:, [-1], :]
        return self.lm_head(logits_input)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        *,
        refinement_steps: Optional[int] = None,
        return_full_logits: Optional[bool] = None,
        output_hidden_states: bool = False,
        return_dict: bool = False,
    ) -> Union[Tuple[torch.Tensor, Optional[torch.Tensor]], TinyLLMOutput]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape (batch, sequence)")
        _, sequence_length = input_ids.shape
        if sequence_length > self.config.block_size:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds block size {self.config.block_size}"
            )
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("input_ids must contain integer token ids")

        steps = self.refinement_steps if refinement_steps is None else refinement_steps
        if steps < 0:
            raise ValueError("refinement_steps cannot be negative")
        if return_full_logits is None:
            return_full_logits = targets is not None

        positions = torch.arange(sequence_length, device=input_ids.device, dtype=torch.long)
        embedded = self.transformer["wte"](input_ids) + self.transformer["wpe"](positions)
        value, states = self._run_blocks(embedded)
        for _ in range(steps):
            value, states = self._run_blocks(embedded, previous_states=states)

        value = self.transformer["ln_f"](value)
        logits_input = value if return_full_logits else value[:, [-1], :]
        logits = self.lm_head(logits_input)
        loss = None
        if targets is not None:
            if targets.shape != input_ids.shape:
                raise ValueError("targets must have the same shape as input_ids")
            if not return_full_logits:
                raise ValueError("targets require return_full_logits=True")
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-1,
            )

        if self.config.capture_activations:
            self.layer_activations = [state.detach() for state in states]
        else:
            self.layer_activations = []
        hidden_states = tuple(states) if output_hidden_states else None
        if return_dict:
            return TinyLLMOutput(logits, loss, hidden_states, steps)
        return logits, loss

    def add_feedback_connection(
        self,
        *,
        num_neurons: Optional[int] = None,
        width: Optional[int] = None,
        source_layer: Optional[int] = None,
        target_layer: Optional[int] = None,
        source_block: Optional[int] = None,
        target_block: Optional[int] = None,
        seed: Optional[int] = None,
        gate_init: Optional[float] = None,
        initial_gate: float = 0.0,
        connection_density: float = 1.0,
        allow_duplicate: bool = False,
    ) -> RecurrentFeedbackConnection:
        """Add a delayed low-rank edge, choosing a reproducible random pair if omitted."""
        if num_neurons is not None and width is not None and num_neurons != width:
            raise ValueError("num_neurons and width disagree")
        resolved_width = num_neurons if num_neurons is not None else width
        if resolved_width is None:
            resolved_width = 16
        if source_layer is not None:
            if source_block is not None and source_layer != source_block:
                raise ValueError("source_layer and source_block disagree")
            source_block = source_layer
        if target_layer is not None:
            if target_block is not None and target_layer != target_block:
                raise ValueError("target_layer and target_block disagree")
            target_block = target_layer
        if gate_init is not None:
            initial_gate = gate_init
        if (source_block is None) != (target_block is None):
            raise ValueError("source_block and target_block must be provided together")

        connection_seed = len(self.feedback_connections) if seed is None else seed
        if source_block is None:
            existing = {
                (connection.source_block, connection.target_block)
                for connection in self.feedback_connections
            }
            candidates = [
                (source, target)
                for source in range(1, self.config.n_layer)
                for target in range(source)
                if allow_duplicate or (source, target) not in existing
            ]
            if not candidates:
                raise ValueError("No unused backward block pairs remain")
            source_block, target_block = random.Random(connection_seed).choice(candidates)

        assert source_block is not None and target_block is not None
        if not 0 <= target_block < source_block < self.config.n_layer:
            raise ValueError(
                "Feedback endpoints must satisfy 0 <= target_block < source_block < n_layer"
            )
        if not allow_duplicate and any(
            connection.source_block == source_block and connection.target_block == target_block
            for connection in self.feedback_connections
        ):
            raise ValueError(f"Feedback connection {source_block}->{target_block} already exists")

        connection = RecurrentFeedbackConnection(
            channels=self.config.n_embd,
            width=resolved_width,
            source_block=source_block,
            target_block=target_block,
            initial_gate=initial_gate,
            connection_density=connection_density,
            seed=connection_seed,
            name=f"Feedback_{len(self.feedback_connections)}_{source_block}_to_{target_block}",
        )
        anchor = self.transformer["wte"].weight
        connection.to(device=anchor.device, dtype=anchor.dtype)
        self.feedback_connections.append(connection)
        self._layers.append(connection)
        # A delayed edge is dormant without a later pass. Make growth active by
        # default while still allowing callers to request zero refinement for a
        # parameter-matched baseline ablation.
        self.refinement_steps = max(1, self.refinement_steps)
        return connection

    def add_random_feedback_connections(
        self,
        *,
        count: int = 1,
        num_neurons: int = 16,
        connection_density: float = 1.0,
        gate_init: float = 0.0,
        seed: int = 0,
    ) -> List[RecurrentFeedbackConnection]:
        """Add reproducible, non-duplicate random backward connections."""
        if count <= 0:
            raise ValueError("count must be positive")
        used_pairs = {
            (connection.source_block, connection.target_block)
            for connection in self.feedback_connections
        }
        available_pairs = self.config.n_layer * (self.config.n_layer - 1) // 2
        remaining_pairs = available_pairs - len(used_pairs)
        if count > remaining_pairs:
            raise ValueError(
                f"Requested {count} random feedback connections but only "
                f"{remaining_pairs} unused backward block pairs remain"
            )
        return [
            self.add_feedback_connection(
                num_neurons=num_neurons,
                connection_density=connection_density,
                gate_init=gate_init,
                seed=seed + index,
            )
            for index in range(count)
        ]

    def get_feedback_topology(self) -> List[Dict[str, Any]]:
        return [connection.describe() for connection in self.feedback_connections]

    def get_layers(self) -> List[ILayer]:
        return list(self._layers)

    def supports_dynamic_growth(self) -> bool:
        return True

    def get_architecture_summary(self) -> Dict[str, Any]:
        feedback_parameters = sum(
            parameter.numel()
            for connection in self.feedback_connections
            for parameter in connection.parameters()
        )
        total_parameters = sum(parameter.numel() for parameter in self.parameters())
        return {
            "family": "TinyLLM GPT-2",
            "preset": self.config.preset,
            "block_size": self.config.block_size,
            "vocab_size": self.config.vocab_size,
            "num_layers": self.config.n_layer,
            "num_heads": self.config.n_head,
            "embedding_width": self.config.n_embd,
            "total_parameters": total_parameters,
            "baseline_parameters": total_parameters - feedback_parameters,
            "feedback_parameters": feedback_parameters,
            "feedback_connections": self.get_feedback_topology(),
            "refinement_steps": self.refinement_steps,
            "llmc_baseline_compatible": len(self.feedback_connections) == 0,
            "feedback_execution": "one-refinement-pass delayed",
        }

    def load_huggingface_state_dict(
        self,
        state_dict: Mapping[str, torch.Tensor],
    ) -> None:
        """Load GPT2LMHeadModel weights, transposing Hugging Face Conv1D tensors."""
        own_state = self.state_dict()
        converted: Dict[str, torch.Tensor] = {}
        missing: List[str] = []
        for key, target in own_state.items():
            if key.startswith("feedback_connections."):
                continue
            if key not in state_dict:
                missing.append(key)
                continue
            source = state_dict[key]
            if key.endswith(self._HF_TRANSPOSED_SUFFIXES):
                source = source.transpose(0, 1)
            if key == "transformer.wte.weight" and source.shape[0] > target.shape[0]:
                source = source[: target.shape[0]]
            if source.shape != target.shape:
                raise ValueError(
                    f"Shape mismatch for {key}: expected {tuple(target.shape)}, got {tuple(source.shape)}"
                )
            converted[key] = source
        if missing:
            raise KeyError(f"Hugging Face state dict is missing: {', '.join(missing)}")
        self.load_state_dict(converted, strict=False)

    def to_huggingface_state_dict(self) -> Dict[str, torch.Tensor]:
        """Export baseline weights in Hugging Face GPT-2 Conv1D layout.

        Feedback execution is not representable by GPT2LMHeadModel or
        llama.cpp, so it must never be silently discarded during export.
        """
        if self.feedback_connections:
            raise RuntimeError(
                "Feedback connections are not representable in GPT-2/Hugging Face/"
                "llama.cpp; distill or remove them before export"
            )
        exported: Dict[str, torch.Tensor] = {}
        for key, value in self.state_dict().items():
            tensor = value.detach().clone()
            if key.endswith(self._HF_TRANSPOSED_SUFFIXES):
                tensor = tensor.transpose(0, 1)
            exported[key] = tensor
        return exported

    def checkpoint_payload(
        self, metadata: Optional[Mapping[str, Any]] = None
    ) -> Dict[str, Any]:
        """Return a self-describing, safely loadable Structure Net checkpoint.

        A raw ``state_dict`` contains feedback masks and weights but not enough
        information to recreate the dynamic modules that own them. This
        envelope stores the model configuration and topology before the tensor
        state so a feedback graph can reconstruct itself exactly.
        """
        safe_metadata = dict(metadata or {})

        def validate_metadata(value: Any, location: str) -> None:
            if value is None or isinstance(value, (str, int, float, bool)):
                return
            if isinstance(value, list):
                for index, item in enumerate(value):
                    validate_metadata(item, f"{location}[{index}]")
                return
            if isinstance(value, dict):
                for key, item in value.items():
                    if not isinstance(key, str):
                        raise TypeError(f"Checkpoint metadata key at {location} must be a string")
                    validate_metadata(item, f"{location}.{key}")
                return
            raise TypeError(
                f"Checkpoint metadata at {location} is not JSON-compatible: "
                f"{type(value).__name__}"
            )

        validate_metadata(safe_metadata, "metadata")
        return {
            "format": self.CHECKPOINT_FORMAT,
            "format_version": self.CHECKPOINT_VERSION,
            "config": asdict(self.config),
            "model_name": self.name,
            "refinement_steps": self.refinement_steps,
            "feedback_topology": self.get_feedback_topology(),
            "state_dict": self.state_dict(),
            "metadata": safe_metadata,
        }

    def save_checkpoint(
        self,
        checkpoint: Union[str, Path],
        *,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Path:
        """Save a topology-aware Structure Net checkpoint."""
        path = Path(checkpoint)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.checkpoint_payload(metadata), path)
        return path

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: Union[str, Path],
        *,
        map_location: Union[str, torch.device] = "cpu",
        name: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "TinyLLMModel":
        """Reconstruct a baseline or feedback graph from its safe envelope."""
        path = Path(checkpoint)
        payload = torch.load(path, map_location=map_location, weights_only=True)
        if not isinstance(payload, dict):
            raise ValueError("TinyLLM checkpoint must contain a dictionary envelope")
        if payload.get("format") != cls.CHECKPOINT_FORMAT:
            raise ValueError(f"Unsupported TinyLLM checkpoint format: {payload.get('format')!r}")
        if payload.get("format_version") != cls.CHECKPOINT_VERSION:
            raise ValueError(
                f"Unsupported TinyLLM checkpoint version: {payload.get('format_version')!r}"
            )
        for required in ("config", "feedback_topology", "state_dict"):
            if required not in payload:
                raise ValueError(f"TinyLLM checkpoint is missing {required!r}")

        if not isinstance(payload["config"], Mapping):
            raise ValueError("TinyLLM checkpoint config must be a mapping")
        if not isinstance(payload["feedback_topology"], list):
            raise ValueError("TinyLLM checkpoint feedback_topology must be a list")
        if not isinstance(payload["state_dict"], Mapping):
            raise ValueError("TinyLLM checkpoint state_dict must be a mapping")

        model = cls(
            TinyLLMConfig(**payload["config"]),
            name=name or payload.get("model_name"),
        )
        for index, topology in enumerate(payload["feedback_topology"]):
            if not isinstance(topology, Mapping):
                raise ValueError(f"Feedback topology entry {index} must be a mapping")
            missing_topology = {
                "source_block",
                "target_block",
            } - set(topology)
            if missing_topology or not ({"width", "num_neurons"} & set(topology)):
                missing = sorted(missing_topology or {"width or num_neurons"})
                raise ValueError(
                    f"Feedback topology entry {index} is missing {', '.join(missing)}"
                )
            model.add_feedback_connection(
                source_block=int(topology["source_block"]),
                target_block=int(topology["target_block"]),
                width=int(
                    topology["width"]
                    if "width" in topology
                    else topology["num_neurons"]
                ),
                connection_density=float(topology.get("connection_density", 1.0)),
                allow_duplicate=True,
            )
        refinement_steps = int(payload.get("refinement_steps", 0))
        if refinement_steps < 0:
            raise ValueError("TinyLLM checkpoint refinement_steps cannot be negative")
        if dtype is None:
            embedding = payload["state_dict"].get("transformer.wte.weight")
            dtype = embedding.dtype if isinstance(embedding, torch.Tensor) else torch.float32
        model.to(device=map_location, dtype=dtype)
        model.load_state_dict(payload["state_dict"], strict=True)
        model.refinement_steps = refinement_steps
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValueError("TinyLLM checkpoint metadata must be a mapping")
        model.checkpoint_metadata = {
            **dict(metadata),
            "path": str(path),
            "format": payload["format"],
            "version": payload["format_version"],
        }
        return model

    @classmethod
    def from_llmc_checkpoint(
        cls,
        checkpoint: Union[str, Path],
        *,
        dtype: torch.dtype = torch.float32,
        device: Union[str, torch.device] = "cpu",
        name: Optional[str] = None,
    ) -> "TinyLLMModel":
        """Load a TinyLLM/llm.c version-3 (fp32) or version-5 (bf16) checkpoint."""
        path = Path(checkpoint)
        with path.open("rb") as handle:
            header = np.fromfile(handle, dtype="<i4", count=256)
            if header.size != 256:
                raise ValueError("Truncated llm.c checkpoint header")
            if int(header[0]) != 20240326:
                raise ValueError("Invalid llm.c checkpoint magic number")
            version = int(header[1])
            if version not in (3, 5):
                raise ValueError(f"Unsupported llm.c checkpoint version {version}")

            block_size, vocab_size, n_layer, n_head, n_embd, padded_vocab = map(
                int, header[2:8]
            )
            if padded_vocab < vocab_size:
                raise ValueError("Padded vocabulary cannot be smaller than vocabulary")
            config = TinyLLMConfig(
                block_size=block_size,
                vocab_size=vocab_size,
                n_layer=n_layer,
                n_head=n_head,
                n_embd=n_embd,
            )
            model = cls(config, name=name).to(device=device, dtype=dtype)
            storage_dtype = np.dtype("<f4") if version == 3 else np.dtype("<u2")

            def read_tensor(shape: Tuple[int, ...]) -> torch.Tensor:
                count = math.prod(shape)
                values = np.fromfile(handle, dtype=storage_dtype, count=count)
                if values.size != count:
                    raise ValueError("Truncated llm.c checkpoint parameters")
                tensor = torch.from_numpy(values.copy())
                if version == 5:
                    tensor = tensor.view(torch.bfloat16)
                return tensor.reshape(shape)

            def copy_parameter(parameter: torch.Tensor, shape: Tuple[int, ...]) -> None:
                value = read_tensor(shape)
                with torch.no_grad():
                    parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))

            channels = config.n_embd
            blocks = model.transformer["h"]
            copy_parameter(model.transformer["wte"].weight, (vocab_size, channels))
            padding_bytes = (padded_vocab - vocab_size) * channels * storage_dtype.itemsize
            handle.seek(padding_bytes, 1)
            copy_parameter(model.transformer["wpe"].weight, (block_size, channels))

            grouped_parameters = (
                (lambda block: block.ln_1.weight, (channels,)),
                (lambda block: block.ln_1.bias, (channels,)),
                (lambda block: block.attn.c_attn.weight, (3 * channels, channels)),
                (lambda block: block.attn.c_attn.bias, (3 * channels,)),
                (lambda block: block.attn.c_proj.weight, (channels, channels)),
                (lambda block: block.attn.c_proj.bias, (channels,)),
                (lambda block: block.ln_2.weight, (channels,)),
                (lambda block: block.ln_2.bias, (channels,)),
                (lambda block: block.mlp.c_fc.weight, (4 * channels, channels)),
                (lambda block: block.mlp.c_fc.bias, (4 * channels,)),
                (lambda block: block.mlp.c_proj.weight, (channels, 4 * channels)),
                (lambda block: block.mlp.c_proj.bias, (channels,)),
            )
            for select_parameter, shape in grouped_parameters:
                for block in blocks:
                    copy_parameter(select_parameter(block), shape)
            copy_parameter(model.transformer["ln_f"].weight, (channels,))
            copy_parameter(model.transformer["ln_f"].bias, (channels,))
            if handle.read(1):
                raise ValueError("Unexpected trailing data in llm.c checkpoint")

        model.checkpoint_metadata = {
            "path": str(path),
            "format": "llm.c",
            "version": version,
            "padded_vocab_size": padded_vocab,
        }
        return model


def create_tinyllm_model(
    preset: Union[str, TinyLLMConfig] = "d6",
    *,
    feedback_connections: int = 0,
    feedback_width: int = 16,
    feedback_connection_density: float = 1.0,
    feedback_seed: int = 0,
    initial_feedback_gate: float = 0.0,
    **config_overrides: Any,
) -> TinyLLMModel:
    """Build a TinyLLM baseline and optionally seed random delayed feedback edges."""
    if isinstance(preset, TinyLLMConfig):
        if config_overrides:
            raise ValueError("config_overrides cannot be used with a TinyLLMConfig instance")
        config = preset
    else:
        config = TinyLLMConfig.from_preset(preset, **config_overrides)
    if feedback_connections < 0:
        raise ValueError("feedback_connections cannot be negative")
    model = TinyLLMModel(config)
    for index in range(feedback_connections):
        model.add_feedback_connection(
            num_neurons=feedback_width,
            connection_density=feedback_connection_density,
            seed=feedback_seed + index,
            initial_gate=initial_feedback_gate,
        )
    if feedback_connections:
        model.refinement_steps = max(1, model.refinement_steps)
    return model


build_tinyllm_model = create_tinyllm_model


__all__ = [
    "TINYLLM_PRESETS",
    "TinyLLMConfig",
    "TinyLLMOutput",
    "TinyLLMBlock",
    "RecurrentFeedbackConnection",
    "BackwardFeedbackPatch",
    "TinyLLMModel",
    "create_tinyllm_model",
    "build_tinyllm_model",
]

"""CALM-style continuous autoregression over a TinyLLM backbone.

This module adapts the two-stage construction from *Continuous Autoregressive
Language Models* to Structure Net's GPT-2-compatible TinyLLM component.  It is
an experimental research implementation: the computation and checkpoint
lifecycle are supported, while pretrained quality retention remains an
empirical question.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import torch
import torch.nn.functional as F
from torch import nn

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
from .tinyllm_model import TinyLLMConfig, TinyLLMModel


@dataclass(frozen=True)
class CALMTinyLLMConfig:
    """Complete configuration for a CALM-style TinyLLM.

    ``backbone.block_size`` counts chunk positions.  The maximum token context
    is therefore ``backbone.block_size * patch_size``.
    """

    backbone: TinyLLMConfig = field(default_factory=TinyLLMConfig)
    patch_size: int = 2
    latent_size: int = 64
    autoencoder_hidden_size: int = 384
    autoencoder_intermediate_size: int = 1024
    autoencoder_dropout: float = 0.15
    kl_weight: float = 1e-3
    kl_floor: float = 0.5
    generator_layers: int = 2
    noise_size: int = 64
    energy_samples: int = 8
    target_samples: int = 100
    energy_beta: float = 1.0
    initialization_seed: int = 42

    def __post_init__(self) -> None:
        positive = {
            "patch_size": self.patch_size,
            "latent_size": self.latent_size,
            "autoencoder_hidden_size": self.autoencoder_hidden_size,
            "autoencoder_intermediate_size": self.autoencoder_intermediate_size,
            "generator_layers": self.generator_layers,
            "noise_size": self.noise_size,
            "energy_samples": self.energy_samples,
            "target_samples": self.target_samples,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.energy_samples < 2:
            raise ValueError("energy_samples must be at least two")
        if not 0.0 <= self.autoencoder_dropout < 1.0:
            raise ValueError("autoencoder_dropout must be in [0, 1)")
        if self.kl_weight < 0.0 or self.kl_floor < 0.0:
            raise ValueError("KL settings cannot be negative")
        if not 0.0 < self.energy_beta < 2.0:
            raise ValueError("energy_beta must be in (0, 2)")

    @property
    def token_block_size(self) -> int:
        return self.backbone.block_size * self.patch_size

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CALMTinyLLMConfig":
        payload = dict(value)
        payload["backbone"] = TinyLLMConfig(**dict(payload["backbone"]))
        return cls(**payload)


@dataclass
class ChunkAutoencoderOutput:
    loss: torch.Tensor
    reconstruction_loss: torch.Tensor
    kl_loss: torch.Tensor
    logits: torch.Tensor
    mean: torch.Tensor
    log_std: torch.Tensor


@dataclass
class CALMTinyLLMOutput:
    loss: torch.Tensor
    latent_predictions: torch.Tensor
    target_mean: torch.Tensor
    target_log_std: torch.Tensor
    hidden_states: Optional[torch.Tensor] = None


class ResidualMLP(BaseLayer):
    """Small pre-normalized residual MLP used by the chunk interfaces."""

    def __init__(self, width: int, intermediate_size: int, name: str):
        super().__init__(name=name)
        self.norm = nn.LayerNorm(width)
        self.fc_in = nn.Linear(width, intermediate_size)
        self.fc_out = nn.Linear(intermediate_size, width)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        hidden = self.fc_out(F.gelu(self.fc_in(self.norm(value)), approximate="tanh"))
        return value + hidden

    def supports_modification(self) -> bool:
        return False


class ChunkEncoder(BaseLayer):
    """Encode fixed-size token chunks as diagonal-Gaussian parameters."""

    def __init__(self, config: CALMTinyLLMConfig):
        super().__init__(name="CALMChunkEncoder")
        width = config.autoencoder_hidden_size
        self.patch_size = config.patch_size
        self.embedding = nn.Embedding(config.backbone.vocab_size, width)
        self.token_mlp = ResidualMLP(
            width, config.autoencoder_intermediate_size, "CALMEncoderTokenMLP"
        )
        self.squeeze = nn.Linear(self.patch_size * width, width)
        self.chunk_mlp = ResidualMLP(
            width, config.autoencoder_intermediate_size, "CALMEncoderChunkMLP"
        )
        self.norm = nn.LayerNorm(width)
        self.to_parameters = nn.Linear(width, 2 * config.latent_size)

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if input_ids.ndim != 2 or input_ids.shape[1] % self.patch_size:
            raise ValueError("input_ids must have shape [batch, tokens divisible by patch_size]")
        batch_size, token_count = input_ids.shape
        chunk_count = token_count // self.patch_size
        chunks = input_ids.reshape(batch_size * chunk_count, self.patch_size)
        hidden = self.token_mlp(self.embedding(chunks))
        hidden = self.squeeze(hidden.reshape(batch_size * chunk_count, -1))
        hidden = self.chunk_mlp(hidden)
        parameters = self.to_parameters(self.norm(hidden))
        parameters = parameters.reshape(batch_size, chunk_count, -1)
        return parameters.chunk(2, dim=-1)

    def supports_modification(self) -> bool:
        return False


class ChunkDecoder(BaseLayer):
    """Decode continuous chunk latents into K independent token logits."""

    def __init__(self, config: CALMTinyLLMConfig, token_weight: nn.Parameter):
        super().__init__(name="CALMChunkDecoder")
        width = config.autoencoder_hidden_size
        self.patch_size = config.patch_size
        self.latent_to_hidden = nn.Linear(config.latent_size, width)
        self.chunk_mlp = ResidualMLP(
            width, config.autoencoder_intermediate_size, "CALMDecoderChunkMLP"
        )
        self.expand = nn.Linear(width, self.patch_size * width)
        self.token_mlp = ResidualMLP(
            width, config.autoencoder_intermediate_size, "CALMDecoderTokenMLP"
        )
        self.norm = nn.LayerNorm(width)
        self.token_weight = token_weight

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim < 2:
            raise ValueError("latent must include a latent feature dimension")
        leading_shape = latent.shape[:-1]
        hidden = self.chunk_mlp(self.latent_to_hidden(latent))
        hidden = self.expand(hidden).reshape(*leading_shape, self.patch_size, -1)
        hidden = self.norm(self.token_mlp(hidden))
        return F.linear(hidden, self.token_weight)

    def supports_modification(self) -> bool:
        return False


class RobustChunkAutoencoder(BaseModel):
    """Variational, dropout-regularized token-chunk autoencoder."""

    def __init__(self, config: CALMTinyLLMConfig):
        super().__init__(name="CALMRobustChunkAutoencoder")
        self.config = config
        self.encoder = ChunkEncoder(config)
        self.decoder = ChunkDecoder(config, self.encoder.embedding.weight)
        self._layers = [self.encoder, self.decoder]
        self._contract = ComponentContract(
            component_name=self.name,
            version=ComponentVersion(1, 0, 0),
            maturity=Maturity.EXPERIMENTAL,
            required_inputs={"batch.input_ids"},
            provided_outputs={
                "autoencoder.loss",
                "autoencoder.reconstruction_loss",
                "autoencoder.kl_loss",
                "autoencoder.latent_distribution",
            },
            resources=ResourceRequirements(
                memory_level=ResourceLevel.MEDIUM,
                requires_gpu=False,
                parallel_safe=True,
            ),
        )

    @property
    def contract(self) -> ComponentContract:
        return self._contract

    def get_layers(self) -> List[ILayer]:
        return list(self._layers)

    def supports_dynamic_growth(self) -> bool:
        return False

    def get_architecture_summary(self) -> Dict[str, Any]:
        return {
            "family": "robust token-chunk variational autoencoder",
            "patch_size": self.config.patch_size,
            "latent_size": self.config.latent_size,
            "hidden_size": self.config.autoencoder_hidden_size,
            "total_parameters": sum(parameter.numel() for parameter in self.parameters()),
            "tied_token_weights": self.decoder.token_weight is self.encoder.embedding.weight,
        }

    def encode_parameters(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(input_ids)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)

    def forward(self, input_ids: torch.Tensor) -> ChunkAutoencoderOutput:
        labels = input_ids
        encoded_ids = input_ids
        if self.training and self.config.autoencoder_dropout:
            keep = torch.rand_like(input_ids, dtype=torch.float32) >= self.config.autoencoder_dropout
            encoded_ids = input_ids * keep.to(input_ids.dtype)

        mean, log_std = self.encode_parameters(encoded_ids)
        if self.training:
            latent = mean + torch.randn_like(mean) * torch.exp(log_std)
            latent = F.dropout(
                latent, p=self.config.autoencoder_dropout, training=True
            )
        else:
            latent = mean
        logits = self.decode(latent)
        reconstruction = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), labels.reshape(-1)
        )
        variance = torch.exp(2.0 * log_std)
        kl_by_dimension = 0.5 * (mean.square() + variance - 1.0 - 2.0 * log_std)
        kl = torch.clamp(kl_by_dimension, min=self.config.kl_floor).sum(dim=-1).mean()
        loss = self.config.patch_size * reconstruction + self.config.kl_weight * kl
        return ChunkAutoencoderOutput(loss, reconstruction, kl, logits, mean, log_std)


class PatchInputAdapter(BaseLayer):
    """Compress K discrete token embeddings into one backbone position."""

    def __init__(self, patch_size: int, width: int):
        super().__init__(name="CALMDiscretePatchInputAdapter")
        self.patch_size = patch_size
        self.projection = nn.Sequential(
            nn.Linear(patch_size * width, 2 * width),
            nn.SiLU(),
            nn.Linear(2 * width, width),
            nn.LayerNorm(width),
        )

    def forward(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        if token_embeddings.ndim != 3 or token_embeddings.shape[1] % self.patch_size:
            raise ValueError("token_embeddings must contain complete token patches")
        batch_size, token_count, width = token_embeddings.shape
        return self.projection(
            token_embeddings.reshape(
                batch_size, token_count // self.patch_size, self.patch_size * width
            )
        )

    def supports_modification(self) -> bool:
        return False


class ConditionalGeneratorBlock(BaseLayer):
    """Residual MLP block that mixes sampling noise with context."""

    def __init__(self, width: int, index: int):
        super().__init__(name=f"CALMConditionalGeneratorBlock_{index}")
        self.norm = nn.LayerNorm(width)
        self.fusion = nn.Sequential(
            nn.Linear(2 * width, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, 2 * width),
        )
        self.down = nn.Linear(width, width)

    def forward(self, value: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        gate, update = self.fusion(torch.cat((self.norm(value), context), dim=-1)).chunk(
            2, dim=-1
        )
        return value + self.down(F.silu(gate) * update)

    def supports_modification(self) -> bool:
        return False


class EnergyBasedGenerativeHead(BaseLayer):
    """Single-step stochastic continuous-vector generator."""

    def __init__(self, config: CALMTinyLLMConfig):
        super().__init__(name="CALMEnergyBasedGenerativeHead")
        width = config.backbone.n_embd
        self.noise_size = config.noise_size
        self.noise_embedding = nn.Linear(config.noise_size, width)
        self.context_embedding = nn.Linear(width, width)
        self.noise_norm = nn.LayerNorm(width)
        self.context_norm = nn.LayerNorm(width)
        self.blocks = nn.ModuleList(
            [ConditionalGeneratorBlock(width, index) for index in range(config.generator_layers)]
        )
        self.output_norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, config.latent_size)

    def forward(
        self,
        context: torch.Tensor,
        *,
        num_samples: int,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        expected_shape = (num_samples, *context.shape[:-1], self.noise_size)
        if noise is None:
            noise = torch.rand(expected_shape, device=context.device, dtype=context.dtype) - 0.5
        elif tuple(noise.shape) != expected_shape:
            raise ValueError(f"noise must have shape {expected_shape}, got {tuple(noise.shape)}")
        expanded_context = context.unsqueeze(0).expand(num_samples, *context.shape)
        value = self.noise_norm(self.noise_embedding(noise))
        conditioned = self.context_norm(self.context_embedding(expanded_context))
        for block in self.blocks:
            value = block(value, conditioned)
        return self.output(self.output_norm(value))

    def supports_modification(self) -> bool:
        return False


class EnergyScoreObjective(BaseLayer):
    """Sample-based proper scoring objective for continuous targets."""

    def __init__(self, beta: float, target_samples: int):
        super().__init__(name="CALMEnergyScoreObjective")
        self.beta = beta
        self.target_samples = target_samples

    def _distance(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return torch.linalg.vector_norm(left - right, dim=-1).pow(self.beta)

    def forward(
        self,
        predictions: torch.Tensor,
        target_mean: torch.Tensor,
        target_log_std: torch.Tensor,
        *,
        target_noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        sample_count = predictions.shape[0]
        if sample_count < 2:
            raise ValueError("energy score requires at least two model samples")
        pairwise = self._distance(predictions[:, None], predictions[None, :])
        off_diagonal = ~torch.eye(
            sample_count, device=predictions.device, dtype=torch.bool
        )
        model_spread = pairwise[off_diagonal].reshape(
            sample_count * (sample_count - 1), *predictions.shape[1:-1]
        ).mean(dim=0)

        expected_target_shape = (
            self.target_samples,
            *target_mean.shape,
        )
        if target_noise is None:
            target_noise = torch.randn(
                expected_target_shape,
                device=target_mean.device,
                dtype=target_mean.dtype,
            )
        elif tuple(target_noise.shape) != expected_target_shape:
            raise ValueError(
                f"target_noise must have shape {expected_target_shape}, "
                f"got {tuple(target_noise.shape)}"
            )
        targets = target_mean.unsqueeze(0) + target_noise * torch.exp(
            target_log_std
        ).unsqueeze(0)
        cross_distance = self._distance(predictions[:, None], targets[None, :]).mean(
            dim=(0, 1)
        )
        return (2.0 * cross_distance - model_spread).mean()

    def supports_modification(self) -> bool:
        return False


class CALMTinyLLMModel(BaseModel):
    """Continuous autoregressive model initialized from a TinyLLM backbone."""

    CHECKPOINT_FORMAT = "structure_net.calm_tinyllm"
    CHECKPOINT_VERSION = 1

    def __init__(self, config: CALMTinyLLMConfig, name: Optional[str] = None):
        super().__init__(name or "CALMTinyLLMModel")
        self.config = config
        self.backbone = TinyLLMModel(config.backbone, name="CALMTinyLLMBackbone")
        self.autoencoder = RobustChunkAutoencoder(config)
        self.input_adapter = PatchInputAdapter(config.patch_size, config.backbone.n_embd)
        self.generative_head = EnergyBasedGenerativeHead(config)
        self.energy_objective = EnergyScoreObjective(
            config.energy_beta, config.target_samples
        )
        self.autoencoder_frozen = False
        self.checkpoint_metadata: Dict[str, Any] = {}
        self._layers = [
            self.autoencoder.encoder,
            self.autoencoder.decoder,
            self.input_adapter,
            *self.backbone.get_layers(),
            self.generative_head,
            self.energy_objective,
        ]
        self._contract = ComponentContract(
            component_name=self.name,
            version=ComponentVersion(1, 0, 0),
            maturity=Maturity.EXPERIMENTAL,
            required_inputs={"batch.input_ids", "autoencoder.checkpoint"},
            optional_inputs={"sampling.noise", "energy.target_noise"},
            provided_outputs={
                "model.energy_loss",
                "model.latent_samples",
                "model.generated_token_chunk",
                "model.architecture",
            },
            resources=ResourceRequirements(
                memory_level=ResourceLevel.HIGH,
                requires_gpu=False,
                parallel_safe=True,
            ),
        )
        self._initialize_new_components()

    def _initialize_new_components(self) -> None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.config.initialization_seed)
        backbone_ids = {id(parameter) for parameter in self.backbone.parameters()}
        for name, parameter in self.named_parameters():
            if id(parameter) in backbone_ids:
                continue
            if parameter.ndim >= 2:
                nn.init.normal_(parameter, mean=0.0, std=0.02, generator=generator)
            elif parameter.ndim == 1:
                if name.endswith(".weight"):
                    nn.init.ones_(parameter)
                else:
                    nn.init.zeros_(parameter)

    @property
    def contract(self) -> ComponentContract:
        return self._contract

    @classmethod
    def from_tinyllm(
        cls,
        source: TinyLLMModel,
        config: CALMTinyLLMConfig,
        *,
        initialize_autoencoder_embedding: bool = True,
    ) -> "CALMTinyLLMModel":
        if source.feedback_connections:
            raise ValueError("CALM conversion requires an ordinary TinyLLM without feedback")
        if source.config != config.backbone:
            raise ValueError("source TinyLLM configuration must match the CALM backbone")
        model = cls(config)
        model.backbone.load_state_dict(source.state_dict(), strict=True)
        if initialize_autoencoder_embedding:
            ae_weight = model.autoencoder.encoder.embedding.weight
            source_weight = source.transformer["wte"].weight
            if ae_weight.shape == source_weight.shape:
                with torch.no_grad():
                    ae_weight.copy_(source_weight)
        return model

    def get_layers(self) -> List[ILayer]:
        return list(self._layers)

    def supports_dynamic_growth(self) -> bool:
        return False

    def set_autoencoder_frozen(self, frozen: bool = True) -> None:
        self.autoencoder_frozen = bool(frozen)
        for parameter in self.autoencoder.parameters():
            parameter.requires_grad = not frozen
        if frozen:
            self.autoencoder.eval()

    def train(self, mode: bool = True) -> "CALMTinyLLMModel":
        super().train(mode)
        if self.autoencoder_frozen:
            self.autoencoder.eval()
        return self

    def _validate_tokens(self, input_ids: torch.Tensor, *, minimum_chunks: int) -> None:
        if input_ids.ndim != 2 or input_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("input_ids must be an integer tensor with shape [batch, tokens]")
        if input_ids.shape[1] % self.config.patch_size:
            raise ValueError("input token count must be divisible by patch_size")
        chunk_count = input_ids.shape[1] // self.config.patch_size
        if chunk_count < minimum_chunks:
            raise ValueError(f"input must contain at least {minimum_chunks} complete chunks")
        if chunk_count > self.config.backbone.block_size:
            raise ValueError("input exceeds the configured CALM chunk context")

    def encode_context(self, input_ids: torch.Tensor) -> torch.Tensor:
        self._validate_tokens(input_ids, minimum_chunks=1)
        chunk_count = input_ids.shape[1] // self.config.patch_size
        token_embeddings = self.backbone.transformer["wte"](input_ids)
        value = self.input_adapter(token_embeddings)
        positions = torch.arange(chunk_count, device=input_ids.device, dtype=torch.long)
        value = value + self.backbone.transformer["wpe"](positions)
        value, _ = self.backbone._run_blocks(value)
        return self.backbone.transformer["ln_f"](value)

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        noise: Optional[torch.Tensor] = None,
        target_noise: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
    ) -> CALMTinyLLMOutput:
        self._validate_tokens(input_ids, minimum_chunks=2)
        patch_size = self.config.patch_size
        batch_size = input_ids.shape[0]
        chunk_count = input_ids.shape[1] // patch_size
        chunks = input_ids.reshape(batch_size, chunk_count, patch_size)
        context_tokens = chunks[:, :-1].reshape(batch_size, -1)
        target_tokens = chunks[:, 1:].reshape(batch_size, -1)
        hidden = self.encode_context(context_tokens)

        with torch.set_grad_enabled(not self.autoencoder_frozen):
            target_mean, target_log_std = self.autoencoder.encode_parameters(target_tokens)
        if self.autoencoder_frozen:
            target_mean = target_mean.detach()
            target_log_std = target_log_std.detach()
        predictions = self.generative_head(
            hidden,
            num_samples=self.config.energy_samples,
            noise=noise,
        )
        loss = self.energy_objective(
            predictions,
            target_mean,
            target_log_std,
            target_noise=target_noise,
        )
        return CALMTinyLLMOutput(
            loss=loss,
            latent_predictions=predictions,
            target_mean=target_mean,
            target_log_std=target_log_std,
            hidden_states=hidden if output_hidden_states else None,
        )

    @torch.no_grad()
    def generate_next_chunk(
        self,
        input_ids: torch.Tensor,
        *,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self.eval()
        hidden = self.encode_context(input_ids)[:, -1, :]
        latent = self.generative_head(hidden, num_samples=1, noise=noise)[0]
        logits = self.autoencoder.decode(latent)
        return logits.argmax(dim=-1)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_chunks: int = 1,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        if max_new_chunks < 0:
            raise ValueError("max_new_chunks cannot be negative")
        generated = input_ids
        for _ in range(max_new_chunks):
            context = generated[:, -self.config.token_block_size :]
            next_chunk = self.generate_next_chunk(context)
            generated = torch.cat((generated, next_chunk), dim=1)
            if eos_token_id is not None and (next_chunk == eos_token_id).any(dim=1).all():
                break
        return generated

    def get_architecture_summary(self) -> Dict[str, Any]:
        autoencoder_parameters = sum(
            parameter.numel() for parameter in self.autoencoder.parameters()
        )
        adapter_parameters = sum(
            parameter.numel() for parameter in self.input_adapter.parameters()
        )
        generator_parameters = sum(
            parameter.numel() for parameter in self.generative_head.parameters()
        )
        return {
            "family": "CALM-style TinyLLM",
            "backbone_family": "TinyLLM GPT-2",
            "patch_size": self.config.patch_size,
            "token_block_size": self.config.token_block_size,
            "chunk_block_size": self.config.backbone.block_size,
            "latent_size": self.config.latent_size,
            "energy_samples": self.config.energy_samples,
            "target_samples": self.config.target_samples,
            "total_parameters": sum(parameter.numel() for parameter in self.parameters()),
            "trainable_parameters": sum(
                parameter.numel() for parameter in self.parameters() if parameter.requires_grad
            ),
            "backbone_parameters": sum(
                parameter.numel() for parameter in self.backbone.parameters()
            ),
            "autoencoder_parameters": autoencoder_parameters,
            "input_adapter_parameters": adapter_parameters,
            "generative_head_parameters": generator_parameters,
            "autoencoder_frozen": self.autoencoder_frozen,
            "source_checkpoint_function_preserved": False,
            "source_backbone_tensors_reusable": True,
        }

    def checkpoint_payload(
        self, metadata: Optional[Mapping[str, Any]] = None
    ) -> Dict[str, Any]:
        safe_metadata = dict(metadata or {})
        json.dumps(safe_metadata, allow_nan=False)
        return {
            "format": self.CHECKPOINT_FORMAT,
            "format_version": self.CHECKPOINT_VERSION,
            "config": self.config.to_dict(),
            "model_name": self.name,
            "autoencoder_frozen": self.autoencoder_frozen,
            "state_dict": self.state_dict(),
            "metadata": safe_metadata,
        }

    def save_checkpoint(
        self,
        checkpoint: Path | str,
        *,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Path:
        path = Path(checkpoint)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.checkpoint_payload(metadata), path)
        return path

    @classmethod
    def from_checkpoint(
        cls, checkpoint: Path | str, *, map_location: Any = "cpu"
    ) -> "CALMTinyLLMModel":
        payload = torch.load(checkpoint, map_location=map_location, weights_only=True)
        if payload.get("format") != cls.CHECKPOINT_FORMAT:
            raise ValueError("not a CALMTinyLLM checkpoint")
        if int(payload.get("format_version", -1)) != cls.CHECKPOINT_VERSION:
            raise ValueError("unsupported CALMTinyLLM checkpoint version")
        config = CALMTinyLLMConfig.from_dict(payload["config"])
        model = cls(config, name=payload.get("model_name"))
        model.load_state_dict(payload["state_dict"], strict=True)
        model.set_autoencoder_frozen(bool(payload.get("autoencoder_frozen", False)))
        model.checkpoint_metadata = dict(payload.get("metadata", {}))
        return model


__all__ = [
    "CALMTinyLLMConfig",
    "CALMTinyLLMModel",
    "CALMTinyLLMOutput",
    "ChunkAutoencoderOutput",
    "EnergyBasedGenerativeHead",
    "EnergyScoreObjective",
    "PatchInputAdapter",
    "RobustChunkAutoencoder",
]

"""
Model components for Structure Net.

Models are complete neural network architectures that implement
the IModel interface and follow the component architecture.
"""

from .minimal_model import MinimalModel
from .fiber_bundle_model import FiberBundleModel, FiberBundleConfig
from .multi_scale_model import MultiScaleModel
from .tinyllm_model import (
    TINYLLM_PRESETS,
    BackwardFeedbackPatch,
    TinyLLMBlock,
    TinyLLMConfig,
    TinyLLMModel,
    TinyLLMOutput,
    build_tinyllm_model,
    create_tinyllm_model,
)
from .calm_tinyllm_model import (
    CALMTinyLLMConfig,
    CALMTinyLLMModel,
    CALMTinyLLMOutput,
    ChunkAutoencoderOutput,
    EnergyBasedGenerativeHead,
    EnergyScoreObjective,
    PatchInputAdapter,
    RobustChunkAutoencoder,
)

__all__ = [
    'MinimalModel',
    'FiberBundleModel',
    'FiberBundleConfig',
    'MultiScaleModel',
    'TINYLLM_PRESETS',
    'TinyLLMConfig',
    'TinyLLMBlock',
    'BackwardFeedbackPatch',
    'TinyLLMModel',
    'TinyLLMOutput',
    'create_tinyllm_model',
    'build_tinyllm_model',
    'CALMTinyLLMConfig',
    'CALMTinyLLMModel',
    'CALMTinyLLMOutput',
    'ChunkAutoencoderOutput',
    'EnergyBasedGenerativeHead',
    'EnergyScoreObjective',
    'PatchInputAdapter',
    'RobustChunkAutoencoder',
]

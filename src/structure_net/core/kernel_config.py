"""
Configuration for the StructureNet kernel.

All kernel-level behavior is declared here rather than scattered across
services. The dataclass is intentionally flat and JSON-serializable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class KernelConfig:
    """Configuration for :class:`structure_net.core.kernel.StructureNetKernel`."""

    # Logging
    log_level: str = "INFO"
    log_handlers: List[str] = field(default_factory=lambda: ["console"])
    log_file: str = "structure_net.log"

    # Profiling
    profile_cpu: bool = True
    profile_memory: bool = True
    profile_gpu: Optional[bool] = None  # None = auto-detect CUDA

    # External services (opt-in; the kernel never requires them)
    use_wandb: bool = False
    wandb_project: str = "structure-net"
    use_chromadb: bool = False
    chromadb_directory: str = "data/chroma_db"

    # Component discovery (import roots that may register components)
    component_packages: List[str] = field(
        default_factory=lambda: ["structure_net.components"]
    )

    def __post_init__(self) -> None:
        valid_handlers = {"console", "file"}
        unknown = set(self.log_handlers) - valid_handlers
        if unknown:
            raise ValueError(f"unknown log handlers: {sorted(unknown)}")
        if self.log_level.upper() not in {
            "DEBUG",
            "INFO",
            "WARNING",
            "ERROR",
            "CRITICAL",
        }:
            raise ValueError(f"unknown log level: {self.log_level}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

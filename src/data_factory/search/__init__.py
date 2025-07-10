"""
ChromaDB Search Layer for Structure Net

Provides semantic search capabilities for experiments, architectures, and hypotheses
using ChromaDB vector database.
"""

from .embedder import (
    ExperimentEmbedder,
    ArchitectureEmbedder,
    embed_experiment,
    embed_architecture,
    embed_hypothesis
)

from .chroma_client import (
    ChromaSearchClient,
    get_chroma_client,
    ChromaConfig
)

from .search_api import (
    ExperimentSearcher,
    search_similar_experiments,
    search_by_architecture,
    search_by_performance,
    search_by_hypothesis
)

__all__ = [
    # Embedders
    'ExperimentEmbedder',
    'ArchitectureEmbedder',
    'embed_experiment',
    'embed_architecture',
    'embed_hypothesis',
    
    # Client
    'ChromaSearchClient',
    'get_chroma_client',
    'ChromaConfig',
    
    # Search API
    'ExperimentSearcher',
    'search_similar_experiments',
    'search_by_architecture',
    'search_by_performance',
    'search_by_hypothesis',
]
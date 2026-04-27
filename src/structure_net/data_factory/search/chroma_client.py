"""
ChromaDB client wrapper for Structure Net.

Manages connections and collections for experiment search.
"""

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import numpy as np
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ChromaConfig:
    """Configuration for ChromaDB client."""
    
    # Storage settings
    persist_directory: str = "data/chroma_db"
    collection_name: str = "structure_net_experiments"
    
    # Client settings
    anonymized_telemetry: bool = False
    allow_reset: bool = False
    
    # Search settings
    default_n_results: int = 10
    max_n_results: int = 100
    
    # Embedding settings
    embedding_dim: int = 384  # Must match embedder output


class ChromaSearchClient:
    """
    Client for managing ChromaDB collections and performing searches.
    
    This class handles:
    - Collection creation and management
    - Document insertion with embeddings
    - Similarity search
    - Metadata filtering
    """
    
    def __init__(self, config: ChromaConfig = None):
        self.config = config or ChromaConfig()
        
        # Ensure persist directory exists
        Path(self.config.persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=self.config.persist_directory,
            settings=Settings(
                anonymized_telemetry=self.config.anonymized_telemetry,
                allow_reset=self.config.allow_reset
            )
        )
        
        # Get or create collection
        self.collection = self._get_or_create_collection()
        
        logger.info(f"ChromaDB client initialized with collection: {self.config.collection_name}")
    
    def _get_or_create_collection(self):
        """Get or create the experiments collection."""
        try:
            collection = self.client.get_collection(
                name=self.config.collection_name
            )
            logger.info(f"Using existing collection: {self.config.collection_name}")
        except ValueError:
            # Collection doesn't exist, create it
            collection = self.client.create_collection(
                name=self.config.collection_name,
                metadata={"description": "Structure Net experiment embeddings"}
            )
            logger.info(f"Created new collection: {self.config.collection_name}")
        
        return collection
    
    def add_experiment(
        self,
        experiment_id: str,
        embedding: np.ndarray,
        metadata: Dict[str, Any],
        document: Optional[str] = None
    ) -> None:
        """
        Add an experiment to the collection.
        
        Args:
            experiment_id: Unique ID for the experiment
            embedding: Vector embedding of the experiment
            metadata: Metadata dict (must contain only str, int, float, bool)
            document: Optional text description
        """
        # Convert numpy array to list
        embedding_list = embedding.tolist()
        
        # Clean metadata (ChromaDB only accepts certain types)
        clean_metadata = self._clean_metadata(metadata)
        
        # Add to collection
        self.collection.add(
            ids=[experiment_id],
            embeddings=[embedding_list],
            metadatas=[clean_metadata],
            documents=[document] if document else None
        )
        
        logger.info(f"Added experiment {experiment_id} to ChromaDB")
    
    def add_experiments_batch(
        self,
        experiment_ids: List[str],
        embeddings: List[np.ndarray],
        metadatas: List[Dict[str, Any]],
        documents: Optional[List[str]] = None
    ) -> None:
        """Add multiple experiments in batch."""
        # Convert embeddings to lists
        embedding_lists = [emb.tolist() for emb in embeddings]
        
        # Clean all metadata
        clean_metadatas = [self._clean_metadata(m) for m in metadatas]
        
        # Add to collection
        self.collection.add(
            ids=experiment_ids,
            embeddings=embedding_lists,
            metadatas=clean_metadatas,
            documents=documents
        )
        
        logger.info(f"Added {len(experiment_ids)} experiments to ChromaDB")
    
    def search_similar(
        self,
        query_embedding: np.ndarray,
        n_results: Optional[int] = None,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Search for similar experiments.
        
        Args:
            query_embedding: Query vector
            n_results: Number of results to return
            where: Metadata filter
            where_document: Document filter
            
        Returns:
            Dictionary with ids, distances, metadatas, documents
        """
        if n_results is None:
            n_results = self.config.default_n_results
        
        n_results = min(n_results, self.config.max_n_results)
        
        # Convert to list
        query_list = query_embedding.tolist()
        
        # Perform search
        results = self.collection.query(
            query_embeddings=[query_list],
            n_results=n_results,
            where=where,
            where_document=where_document
        )
        
        # Flatten results (since we only have one query)
        return {
            'ids': results['ids'][0] if results['ids'] else [],
            'distances': results['distances'][0] if results['distances'] else [],
            'metadatas': results['metadatas'][0] if results['metadatas'] else [],
            'documents': results['documents'][0] if results['documents'] else []
        }
    
    def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific experiment by ID."""
        try:
            result = self.collection.get(
                ids=[experiment_id],
                include=['metadatas', 'documents', 'embeddings']
            )
            
            if result['ids']:
                return {
                    'id': result['ids'][0],
                    'metadata': result['metadatas'][0] if result['metadatas'] else {},
                    'document': result['documents'][0] if result['documents'] else None,
                    'embedding': np.array(result['embeddings'][0]) if result['embeddings'] else None
                }
            return None
        except Exception as e:
            logger.error(f"Error getting experiment {experiment_id}: {e}")
            return None
    
    def update_experiment(
        self,
        experiment_id: str,
        embedding: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None,
        document: Optional[str] = None
    ) -> None:
        """Update an existing experiment."""
        update_kwargs = {'ids': [experiment_id]}
        
        if embedding is not None:
            update_kwargs['embeddings'] = [embedding.tolist()]
        
        if metadata is not None:
            update_kwargs['metadatas'] = [self._clean_metadata(metadata)]
        
        if document is not None:
            update_kwargs['documents'] = [document]
        
        self.collection.update(**update_kwargs)
        logger.info(f"Updated experiment {experiment_id}")
    
    def delete_experiment(self, experiment_id: str) -> None:
        """Delete an experiment from the collection."""
        self.collection.delete(ids=[experiment_id])
        logger.info(f"Deleted experiment {experiment_id}")
    
    def count(self) -> int:
        """Get the number of experiments in the collection."""
        return self.collection.count()
    
    def reset_collection(self) -> None:
        """Reset the collection (delete all data)."""
        if self.config.allow_reset:
            self.client.delete_collection(self.config.collection_name)
            self.collection = self._get_or_create_collection()
            logger.warning("Collection reset - all data deleted")
        else:
            raise ValueError("Collection reset not allowed. Set allow_reset=True in config.")
    
    def _clean_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean metadata to only include types supported by ChromaDB.
        
        ChromaDB only supports: str, int, float, bool
        """
        clean = {}
        
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                clean[key] = value
            elif isinstance(value, list) and all(isinstance(v, (str, int, float, bool)) for v in value):
                # ChromaDB doesn't support lists directly, so we join them
                clean[key] = str(value)
            elif value is None:
                clean[key] = "null"
            else:
                # Convert to string representation
                clean[key] = str(value)
        
        return clean


# Global client instance
_chroma_client: Optional[ChromaSearchClient] = None


def get_chroma_client(config: ChromaConfig = None) -> ChromaSearchClient:
    """Get or create the global ChromaDB client."""
    global _chroma_client
    
    if _chroma_client is None:
        _chroma_client = ChromaSearchClient(config)
    
    return _chroma_client


def reset_chroma_client() -> None:
    """Reset the global ChromaDB client."""
    global _chroma_client
    _chroma_client = None
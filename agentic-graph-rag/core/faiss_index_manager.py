#!/usr/bin/env python3
"""
FAISS Index Manager for Agentic Graph RAG
Manages vector indices for fast similarity search and retrieval operations.
"""

import logging
import numpy as np
import pickle
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
import asyncio
from datetime import datetime

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("FAISS not available. Install with: pip install faiss-cpu")

logger = logging.getLogger(__name__)


@dataclass
class VectorSearchResult:
    """Result from vector similarity search."""
    id: str
    score: float
    metadata: Dict[str, Any]
    content: str
    embedding: Optional[List[float]] = None


@dataclass
class IndexConfig:
    """Configuration for FAISS index."""
    dimension: int = 1536  # OpenAI ada-002 embedding dimension
    index_type: str = "flat"  # flat, ivf, hnsw
    metric_type: str = "cosine"  # cosine, l2, inner_product
    nlist: int = 100  # For IVF indices
    m: int = 16  # For HNSW indices
    ef_construction: int = 200  # For HNSW indices
    ef_search: int = 50  # For HNSW indices


class FAISSIndexManager:
    """
    Manages FAISS vector indices for fast similarity search.
    
    Features:
    - Multiple index types (Flat, IVF, HNSW)
    - Batch operations for efficiency
    - Persistent storage and loading
    - Metadata management
    - Async-compatible operations
    - Graph embedding integration
    """
    
    def __init__(self, config: IndexConfig, index_dir: Union[str, Path] = "data/indexes"):
        """Initialize FAISS index manager."""
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS is required. Install with: pip install faiss-cpu")
        
        self.config = config
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize index
        self.index: Optional[faiss.Index] = None
        self.metadata_store: Dict[int, Dict[str, Any]] = {}
        self.id_to_idx: Dict[str, int] = {}
        self.idx_to_id: Dict[int, str] = {}
        self.next_idx = 0
        
        # Index file paths
        self.index_file = self.index_dir / "faiss_index.bin"
        self.metadata_file = self.index_dir / "metadata.json"
        self.mapping_file = self.index_dir / "id_mapping.json"
        
        self._create_or_load_index()
        logger.info(f"FAISS Index Manager initialized (dimension: {config.dimension})")
    
    def _create_index(self) -> faiss.Index:
        """Create a new FAISS index based on configuration."""
        dimension = self.config.dimension
        
        if self.config.index_type.lower() == "flat":
            # Flat index - exact search
            if self.config.metric_type == "cosine":
                index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            elif self.config.metric_type == "l2":
                index = faiss.IndexFlatL2(dimension)
            else:
                index = faiss.IndexFlatIP(dimension)
        
        elif self.config.index_type.lower() == "ivf":
            # IVF index - approximate search with clustering
            quantizer = faiss.IndexFlatIP(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, self.config.nlist)
        
        elif self.config.index_type.lower() == "hnsw":
            # HNSW index - hierarchical navigable small world
            index = faiss.IndexHNSWFlat(dimension, self.config.m)
            index.hnsw.efConstruction = self.config.ef_construction
            index.hnsw.efSearch = self.config.ef_search
        
        else:
            logger.warning(f"Unknown index type: {self.config.index_type}, using flat")
            index = faiss.IndexFlatIP(dimension)
        
        return index
    
    def _create_or_load_index(self):
        """Create new index or load existing one."""
        if self.index_file.exists():
            self.load_index()
        else:
            self.index = self._create_index()
            logger.info(f"Created new {self.config.index_type} index")
    
    def save_index(self):
        """Save index and metadata to disk."""
        if self.index is not None:
            # Save FAISS index
            faiss.write_index(self.index, str(self.index_file))
            
            # Save metadata
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata_store, f, indent=2, default=str)
            
            # Save ID mappings
            mappings = {
                'id_to_idx': self.id_to_idx,
                'idx_to_id': {str(k): v for k, v in self.idx_to_id.items()},
                'next_idx': self.next_idx
            }
            with open(self.mapping_file, 'w', encoding='utf-8') as f:
                json.dump(mappings, f, indent=2)
            
            logger.info(f"Saved index with {self.index.ntotal} vectors")
    
    def load_index(self):
        """Load index and metadata from disk."""
        try:
            # Load FAISS index
            self.index = faiss.read_index(str(self.index_file))
            
            # Load metadata
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    self.metadata_store = json.load(f)
                    # Convert string keys back to int
                    self.metadata_store = {int(k): v for k, v in self.metadata_store.items()}
            
            # Load ID mappings
            if self.mapping_file.exists():
                with open(self.mapping_file, 'r', encoding='utf-8') as f:
                    mappings = json.load(f)
                    self.id_to_idx = mappings.get('id_to_idx', {})
                    self.idx_to_id = {int(k): v for k, v in mappings.get('idx_to_id', {}).items()}
                    self.next_idx = mappings.get('next_idx', 0)
            
            logger.info(f"Loaded index with {self.index.ntotal} vectors")
        
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            self.index = self._create_index()
    
    def normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalize embeddings for cosine similarity."""
        if self.config.metric_type == "cosine":
            # Normalize for cosine similarity using inner product
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            return embeddings / norms
        return embeddings
    
    async def add_embeddings(self, 
                           embeddings: List[List[float]], 
                           ids: List[str], 
                           metadata: List[Dict[str, Any]]) -> bool:
        """
        Add embeddings to the index.
        
        Args:
            embeddings: List of embedding vectors
            ids: Corresponding unique IDs
            metadata: Associated metadata for each embedding
            
        Returns:
            Success status
        """
        try:
            if len(embeddings) != len(ids) or len(embeddings) != len(metadata):
                raise ValueError("Embeddings, IDs, and metadata must have same length")
            
            # Convert to numpy array
            embeddings_array = np.array(embeddings, dtype=np.float32)
            
            # Normalize if needed
            embeddings_array = self.normalize_embeddings(embeddings_array)
            
            # Train index if needed (for IVF)
            if hasattr(self.index, 'is_trained') and not self.index.is_trained:
                if embeddings_array.shape[0] >= self.config.nlist:
                    self.index.train(embeddings_array)
                    logger.info("Index trained")
                else:
                    logger.warning("Not enough data to train index")
            
            # Add to index
            start_idx = self.next_idx
            self.index.add(embeddings_array)
            
            # Update mappings and metadata
            for i, (id_str, meta) in enumerate(zip(ids, metadata)):
                idx = start_idx + i
                self.id_to_idx[id_str] = idx
                self.idx_to_id[idx] = id_str
                self.metadata_store[idx] = {
                    **meta,
                    'id': id_str,
                    'added_at': datetime.now().isoformat(),
                    'embedding_dim': len(embeddings[i])
                }
            
            self.next_idx += len(embeddings)
            
            logger.info(f"Added {len(embeddings)} embeddings to index")
            return True
        
        except Exception as e:
            logger.error(f"Failed to add embeddings: {e}")
            return False
    
    async def search_similar(self, 
                           query_embedding: List[float], 
                           k: int = 10,
                           threshold: Optional[float] = None) -> List[VectorSearchResult]:
        """
        Search for similar vectors.
        
        Args:
            query_embedding: Query vector
            k: Number of results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of search results
        """
        try:
            if self.index is None or self.index.ntotal == 0:
                return []
            
            # Prepare query
            query_array = np.array([query_embedding], dtype=np.float32)
            query_array = self.normalize_embeddings(query_array)
            
            # Search
            scores, indices = self.index.search(query_array, k)
            
            # Convert results
            results = []
            for i in range(len(indices[0])):
                idx = indices[0][i]
                score = float(scores[0][i])
                
                if idx == -1:  # FAISS returns -1 for empty results
                    continue
                
                if threshold is not None and score < threshold:
                    continue
                
                # Get metadata
                metadata = self.metadata_store.get(idx, {})
                id_str = self.idx_to_id.get(idx, f"unknown_{idx}")
                
                result = VectorSearchResult(
                    id=id_str,
                    score=score,
                    metadata=metadata,
                    content=metadata.get('content', ''),
                    embedding=None  # Don't return embedding by default
                )
                results.append(result)
            
            logger.debug(f"Found {len(results)} similar vectors")
            return results
        
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    async def search_by_ids(self, ids: List[str]) -> List[VectorSearchResult]:
        """Search for vectors by their IDs."""
        results = []
        
        for id_str in ids:
            if id_str in self.id_to_idx:
                idx = self.id_to_idx[id_str]
                metadata = self.metadata_store.get(idx, {})
                
                result = VectorSearchResult(
                    id=id_str,
                    score=1.0,  # Perfect match
                    metadata=metadata,
                    content=metadata.get('content', '')
                )
                results.append(result)
        
        return results
    
    async def remove_embeddings(self, ids: List[str]) -> int:
        """
        Remove embeddings by ID.
        Note: FAISS doesn't support direct removal, so we mark as deleted in metadata.
        """
        removed_count = 0
        
        for id_str in ids:
            if id_str in self.id_to_idx:
                idx = self.id_to_idx[id_str]
                if idx in self.metadata_store:
                    self.metadata_store[idx]['deleted'] = True
                    removed_count += 1
        
        logger.info(f"Marked {removed_count} embeddings as deleted")
        return removed_count
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get index statistics."""
        active_count = sum(1 for meta in self.metadata_store.values() 
                          if not meta.get('deleted', False))
        
        return {
            'total_vectors': self.index.ntotal if self.index else 0,
            'active_vectors': active_count,
            'deleted_vectors': len(self.metadata_store) - active_count,
            'dimension': self.config.dimension,
            'index_type': self.config.index_type,
            'metric_type': self.config.metric_type,
            'memory_usage_mb': self.index.ntotal * self.config.dimension * 4 / 1024 / 1024 if self.index else 0
        }
    
    async def rebuild_index(self) -> bool:
        """Rebuild index excluding deleted entries."""
        try:
            if not self.metadata_store:
                return True
            
            # Collect active embeddings
            active_embeddings = []
            active_ids = []
            active_metadata = []
            
            # We can't get embeddings back from FAISS easily, so we need them stored in metadata
            for idx, meta in self.metadata_store.items():
                if not meta.get('deleted', False) and 'embedding' in meta:
                    active_embeddings.append(meta['embedding'])
                    active_ids.append(meta['id'])
                    active_metadata.append(meta)
            
            if not active_embeddings:
                logger.info("No active embeddings to rebuild")
                return True
            
            # Create new index
            old_index = self.index
            self.index = self._create_index()
            self.metadata_store.clear()
            self.id_to_idx.clear()
            self.idx_to_id.clear()
            self.next_idx = 0
            
            # Add active embeddings
            success = await self.add_embeddings(active_embeddings, active_ids, active_metadata)
            
            if success:
                logger.info(f"Rebuilt index with {len(active_embeddings)} active vectors")
                return True
            else:
                # Restore old index on failure
                self.index = old_index
                return False
        
        except Exception as e:
            logger.error(f"Index rebuild failed: {e}")
            return False
    
    def close(self):
        """Clean up resources."""
        self.save_index()
        if hasattr(self, 'index') and self.index:
            # FAISS indices don't need explicit cleanup in Python
            pass
        logger.info("FAISS Index Manager closed")


# Convenience functions
async def create_index_manager(dimension: int = 1536, 
                             index_type: str = "flat",
                             index_dir: str = "data/indexes") -> FAISSIndexManager:
    """Create a FAISS index manager with default settings."""
    config = IndexConfig(
        dimension=dimension,
        index_type=index_type
    )
    return FAISSIndexManager(config, index_dir)


async def search_graph_embeddings(index_manager: FAISSIndexManager,
                                query_text: str,
                                embedding_function,
                                k: int = 10) -> List[VectorSearchResult]:
    """Search graph embeddings using text query."""
    # Generate embedding for query
    query_embedding = await embedding_function([query_text])
    if not query_embedding:
        return []
    
    # Search similar embeddings
    results = await index_manager.search_similar(query_embedding[0], k=k)
    return results
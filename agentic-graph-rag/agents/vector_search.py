"""
Vector search agent with FAISS-based semantic similarity search.

This module provides semantic search capabilities using vector embeddings
and FAISS (Facebook AI Similarity Search) for efficient nearest neighbor queries.
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import pickle
import json
import numpy as np
from dataclasses import dataclass, asdict

try:
    import faiss  # type: ignore
except ImportError:
    faiss = None

from core.embedding_manager_v2 import EmbeddingManager
from core.graph_store import GraphStore, GraphNode, GraphQuery, QueryResult

logger = logging.getLogger(__name__)


@dataclass
class VectorSearchResult:
    """Result of a vector similarity search."""
    node_id: str
    node: GraphNode
    similarity_score: float
    distance: float
    rank: int


@dataclass
class VectorSearchQuery:
    """Query for vector similarity search."""
    query_text: str
    query_embedding: Optional[List[float]] = None
    top_k: int = 10
    similarity_threshold: float = 0.7
    node_type_filter: Optional[str] = None
    property_filters: Optional[Dict[str, Any]] = None


@dataclass
class VectorIndexStats:
    """Statistics about the vector index."""
    total_vectors: int
    index_dimension: int
    index_type: str
    index_size_mb: float
    last_updated: float
    metadata: Dict[str, Any]


class VectorSearchAgent:
    """
    Provides semantic vector search capabilities using FAISS indexing.
    
    This agent maintains a FAISS index of node embeddings and provides
    efficient similarity search, with automatic index management and
    metadata persistence.
    """
    
    def __init__(self,
                 embedding_manager: EmbeddingManager,
                 graph_store: GraphStore,
                 index_dir: str = "data/graphs/faiss",
                 index_type: str = "IndexFlatIP",  # Inner product (cosine similarity)
                 rebuild_threshold: int = 1000):
        """
        Initialize the vector search agent.
        
        Args:
            embedding_manager: Service for generating embeddings
            graph_store: Graph storage backend
            index_dir: Directory to store FAISS indexes
            index_type: Type of FAISS index to use
            rebuild_threshold: Number of changes before rebuilding index
        """
        if faiss is None:
            raise ImportError("FAISS library is required for vector search. Install with: pip install faiss-cpu")
        
        self.embedding_manager = embedding_manager
        self.graph_store = graph_store
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.index_type = index_type
        self.rebuild_threshold = rebuild_threshold
        
        # FAISS index and metadata
        self.index = None  # Optional[faiss.Index]
        self.node_id_to_index: Dict[str, int] = {}  # Map node IDs to index positions
        self.index_to_node_id: Dict[int, str] = {}  # Map index positions to node IDs
        self.node_metadata: Dict[str, Dict[str, Any]] = {}  # Additional node metadata
        
        # Index files
        self.index_file = self.index_dir / "vector_index.faiss"
        self.metadata_file = self.index_dir / "index_metadata.json"
        
        # Track changes for incremental updates
        self.changes_since_rebuild = 0
    
    async def initialize_index(self, force_rebuild: bool = False) -> bool:
        """
        Initialize or load the FAISS index.
        
        Args:
            force_rebuild: Whether to force rebuilding the index from scratch
            
        Returns:
            bool: True if initialization successful
        """
        try:
            if not force_rebuild and await self._load_existing_index():
                logger.info("Loaded existing FAISS index")
                return True
            else:
                logger.info("Building new FAISS index from graph")
                return await self._build_index_from_graph()
                
        except Exception as e:
            logger.error(f"Failed to initialize vector index: {e}")
            return False
    
    async def _load_existing_index(self) -> bool:
        """Load existing FAISS index from disk."""
        try:
            if not self.index_file.exists() or not self.metadata_file.exists():
                return False
            
            # Load FAISS index
            self.index = faiss.read_index(str(self.index_file))
            
            # Load metadata
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
            
            self.node_id_to_index = metadata.get("node_id_to_index", {})
            self.index_to_node_id = {int(k): v for k, v in metadata.get("index_to_node_id", {}).items()}
            self.node_metadata = metadata.get("node_metadata", {})
            self.changes_since_rebuild = metadata.get("changes_since_rebuild", 0)
            
            logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to load existing index: {e}")
            return False
    
    async def _build_index_from_graph(self) -> bool:
        """Build FAISS index from all nodes in the graph."""
        try:
            # Get embedding dimension
            dimension = await self.embedding_manager.get_embedding_dimension()
            
            # Create FAISS index based on type
            if self.index_type == "IndexFlatIP":
                self.index = faiss.IndexFlatIP(dimension)
            elif self.index_type == "IndexFlatL2":
                self.index = faiss.IndexFlatL2(dimension)
            else:
                raise ValueError(f"Unsupported index type: {self.index_type}")
            
            # Query all nodes with embeddings from graph
            query = GraphQuery(
                cypher="MATCH (n) WHERE n.embedding IS NOT NULL RETURN n, elementId(n) as node_id",
                limit=100000
            )
            
            result = await self.graph_store.query(query)
            
            if not result.nodes:
                logger.warning("No nodes with embeddings found in graph")
                return True
            
            # Prepare vectors and metadata
            vectors = []
            node_ids = []
            
            for i, node in enumerate(result.nodes):
                if node.embedding:
                    vectors.append(node.embedding)
                    node_ids.append(node.id)
                    
                    # Store mapping
                    self.node_id_to_index[node.id] = i
                    self.index_to_node_id[i] = node.id
                    
                    # Store node metadata
                    self.node_metadata[node.id] = {
                        "name": node.name,
                        "type": node.type.value,
                        "properties": node.properties
                    }
            
            if vectors:
                # Convert to numpy array and add to index
                vectors_array = np.array(vectors, dtype=np.float32)
                
                # Normalize for cosine similarity if using inner product
                if self.index_type == "IndexFlatIP":
                    faiss.normalize_L2(vectors_array)
                
                self.index.add(vectors_array)
                
                logger.info(f"Built FAISS index with {len(vectors)} vectors")
            
            # Save index and metadata
            await self._save_index()
            self.changes_since_rebuild = 0
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to build index from graph: {e}")
            return False
    
    async def _save_index(self) -> None:
        """Save FAISS index and metadata to disk."""
        try:
            # Save FAISS index
            faiss.write_index(self.index, str(self.index_file))
            
            # Save metadata
            metadata = {
                "node_id_to_index": self.node_id_to_index,
                "index_to_node_id": {str(k): v for k, v in self.index_to_node_id.items()},
                "node_metadata": self.node_metadata,
                "changes_since_rebuild": self.changes_since_rebuild,
                "index_type": self.index_type,
                "dimension": self.index.d if self.index else 0,
                "total_vectors": self.index.ntotal if self.index else 0,
                "last_updated": asyncio.get_event_loop().time()
            }
            
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
    
    async def search_similar_nodes(self, query: VectorSearchQuery) -> List[VectorSearchResult]:
        """
        Search for nodes similar to the query.
        
        Args:
            query: Vector search query parameters
            
        Returns:
            List[VectorSearchResult]: List of similar nodes with scores
        """
        if not self.index or self.index.ntotal == 0:
            logger.warning("No vector index available")
            return []
        
        try:
            # Get query embedding
            if query.query_embedding:
                query_vector = np.array([query.query_embedding], dtype=np.float32)
            else:
                embedding = await self.embedding_manager.generate_embedding(query.query_text)
                query_vector = np.array([embedding], dtype=np.float32)
            
            # Normalize for cosine similarity if using inner product
            if self.index_type == "IndexFlatIP":
                faiss.normalize_L2(query_vector)
            
            # Perform search
            distances, indices = self.index.search(query_vector, query.top_k)
            
            # Process results
            results = []
            for rank, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx == -1:  # No more results
                    break
                
                node_id = self.index_to_node_id.get(idx)
                if not node_id:
                    continue
                
                # Calculate similarity score (convert from distance)
                if self.index_type == "IndexFlatIP":
                    similarity_score = float(distance)  # Inner product is already similarity
                else:  # L2 distance
                    similarity_score = 1.0 / (1.0 + float(distance))
                
                # Apply similarity threshold
                if similarity_score < query.similarity_threshold:
                    break
                
                # Get full node data
                node = await self.graph_store.get_node(node_id)
                if not node:
                    continue
                
                # Apply filters
                if query.node_type_filter and node.type.value != query.node_type_filter:
                    continue
                
                if query.property_filters:
                    if not self._matches_property_filters(node, query.property_filters):
                        continue
                
                results.append(VectorSearchResult(
                    node_id=node_id,
                    node=node,
                    similarity_score=similarity_score,
                    distance=float(distance),
                    rank=rank
                ))
            
            logger.debug(f"Vector search returned {len(results)} results for query: {query.query_text[:100]}")
            return results
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
    
    def _matches_property_filters(self, node: GraphNode, filters: Dict[str, Any]) -> bool:
        """Check if node matches property filters."""
        for key, expected_value in filters.items():
            node_value = node.properties.get(key)
            
            if isinstance(expected_value, list):
                # Value should be in the list
                if node_value not in expected_value:
                    return False
            elif isinstance(expected_value, dict):
                # Range or complex filter
                if "min" in expected_value and node_value < expected_value["min"]:
                    return False
                if "max" in expected_value and node_value > expected_value["max"]:
                    return False
            else:
                # Exact match
                if node_value != expected_value:
                    return False
        
        return True
    
    async def add_node_to_index(self, node_id: str, embedding: List[float]) -> bool:
        """
        Add a new node to the vector index.
        
        Args:
            node_id: ID of the node to add
            embedding: Vector embedding for the node
            
        Returns:
            bool: True if addition successful
        """
        try:
            if not self.index:
                logger.warning("Index not initialized")
                return False
            
            # Get node metadata
            node = await self.graph_store.get_node(node_id)
            if not node:
                logger.warning(f"Node {node_id} not found in graph")
                return False
            
            # Prepare vector
            vector = np.array([embedding], dtype=np.float32)
            
            # Normalize for cosine similarity if using inner product
            if self.index_type == "IndexFlatIP":
                faiss.normalize_L2(vector)
            
            # Add to index
            current_index = self.index.ntotal
            self.index.add(vector)
            
            # Update mappings
            self.node_id_to_index[node_id] = current_index
            self.index_to_node_id[current_index] = node_id
            
            # Store node metadata
            self.node_metadata[node_id] = {
                "name": node.name,
                "type": node.type.value,
                "properties": node.properties
            }
            
            self.changes_since_rebuild += 1
            
            # Check if rebuild is needed
            if self.changes_since_rebuild >= self.rebuild_threshold:
                logger.info("Rebuild threshold reached, scheduling index rebuild")
                # Note: In a real system, this might be done asynchronously
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to add node to index: {e}")
            return False
    
    async def remove_node_from_index(self, node_id: str) -> bool:
        """
        Remove a node from the vector index.
        
        Note: FAISS doesn't support efficient removal, so this marks for rebuild.
        
        Args:
            node_id: ID of the node to remove
            
        Returns:
            bool: True if removal successful
        """
        try:
            if node_id not in self.node_id_to_index:
                return True  # Already not in index
            
            # Mark for rebuild (FAISS doesn't support efficient removal)
            del self.node_id_to_index[node_id]
            del self.node_metadata[node_id]
            
            # Remove from index_to_node_id mapping
            index_pos = None
            for pos, nid in self.index_to_node_id.items():
                if nid == node_id:
                    index_pos = pos
                    break
            
            if index_pos is not None:
                del self.index_to_node_id[index_pos]
            
            self.changes_since_rebuild += 1
            
            # If too many changes, rebuild
            if self.changes_since_rebuild >= self.rebuild_threshold:
                await self._build_index_from_graph()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove node from index: {e}")
            return False
    
    async def get_index_stats(self) -> VectorIndexStats:
        """Get statistics about the current vector index."""
        index_size_mb = 0.0
        
        try:
            if self.index_file.exists():
                index_size_mb = self.index_file.stat().st_size / 1024 / 1024
        except:
            pass
        
        return VectorIndexStats(
            total_vectors=self.index.ntotal if self.index else 0,
            index_dimension=self.index.d if self.index else 0,
            index_type=self.index_type,
            index_size_mb=index_size_mb,
            last_updated=asyncio.get_event_loop().time(),
            metadata={
                "changes_since_rebuild": self.changes_since_rebuild,
                "rebuild_threshold": self.rebuild_threshold,
                "node_metadata_count": len(self.node_metadata)
            }
        )
    
    async def search_by_text(self, query_text: str, 
                           top_k: int = 10,
                           similarity_threshold: float = 0.7,
                           node_type_filter: Optional[str] = None) -> List[VectorSearchResult]:
        """
        Convenience method for text-based semantic search.
        
        Args:
            query_text: Text to search for
            top_k: Maximum number of results
            similarity_threshold: Minimum similarity score
            node_type_filter: Optional node type filter
            
        Returns:
            List[VectorSearchResult]: Search results
        """
        query = VectorSearchQuery(
            query_text=query_text,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            node_type_filter=node_type_filter
        )
        
        return await self.search_similar_nodes(query)
    
    async def find_related_concepts(self, node_id: str, 
                                  top_k: int = 5,
                                  exclude_self: bool = True) -> List[VectorSearchResult]:
        """
        Find concepts related to a specific node.
        
        Args:
            node_id: ID of the node to find related concepts for
            top_k: Maximum number of results
            exclude_self: Whether to exclude the original node from results
            
        Returns:
            List[VectorSearchResult]: Related concepts
        """
        try:
            # Get the node and its embedding
            node = await self.graph_store.get_node(node_id)
            if not node or not node.embedding:
                return []
            
            # Search for similar nodes
            query = VectorSearchQuery(
                query_text="",  # Not used when embedding is provided
                query_embedding=node.embedding,
                top_k=top_k + (1 if exclude_self else 0),
                similarity_threshold=0.5
            )
            
            results = await self.search_similar_nodes(query)
            
            # Filter out self if requested
            if exclude_self:
                results = [r for r in results if r.node_id != node_id][:top_k]
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to find related concepts for node {node_id}: {e}")
            return []

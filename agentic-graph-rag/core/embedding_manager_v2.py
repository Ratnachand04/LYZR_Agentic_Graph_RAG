#!/usr/bin/env python3
"""
Embedding Manager for Agentic Graph RAG Pipeline
Handles vector embeddings for entities, relationships, and semantic similarity operations.
"""

import asyncio
import logging
import os
from typing import Dict, List, Any, Optional, Tuple
import json
import numpy as np
from dataclasses import dataclass
from datetime import datetime

import httpx
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
import pickle

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingConfig:
    """Configuration for embedding operations."""
    api_key: str
    base_url: str = "https://api.openai.com/v1"
    model: str = "text-embedding-ada-002"
    batch_size: int = 100
    max_retries: int = 3
    timeout: int = 30
    cache_embeddings: bool = True
    cache_file: str = "embeddings_cache.pkl"


class EmbeddingManager:
    """
    Comprehensive embedding management for knowledge graphs.
    
    Features:
    - OpenAI API integration for text embeddings
    - Batch processing for efficiency
    - Embedding cache for performance
    - Similarity search and clustering
    - Entity resolution via embedding similarity
    - Vector operations and analytics
    """
    
    def __init__(self, config: EmbeddingConfig):
        """Initialize embedding manager."""
        self.config = config
        self.client = httpx.AsyncClient(
            base_url=config.base_url,
            headers={
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json"
            },
            timeout=config.timeout
        )
        
        # Embedding cache
        self.embedding_cache: Dict[str, List[float]] = {}
        self.cache_file = config.cache_file
        
        # Load existing cache
        if config.cache_embeddings and os.path.exists(self.cache_file):
            self._load_cache()
        
        logger.info(f"Embedding Manager initialized with model: {config.model}")
    
    async def generate_embeddings(self, 
                                texts: List[str], 
                                cache_keys: Optional[List[str]] = None) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            cache_keys: Optional cache keys for each text
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        logger.info(f"Generating embeddings for {len(texts)} texts")
        
        # Check cache first
        embeddings = []
        texts_to_process = []
        indices_to_process = []
        
        for i, text in enumerate(texts):
            cache_key = cache_keys[i] if cache_keys and i < len(cache_keys) else None
            
            if cache_key and cache_key in self.embedding_cache:
                embeddings.append(self.embedding_cache[cache_key])
            elif text in self.embedding_cache:
                embeddings.append(self.embedding_cache[text])
            else:
                embeddings.append(None)  # Placeholder
                texts_to_process.append(text)
                indices_to_process.append(i)
        
        if texts_to_process:
            logger.info(f"Processing {len(texts_to_process)} uncached texts")
            
            # Process in batches
            new_embeddings = await self._process_embedding_batches(texts_to_process)
            
            # Insert new embeddings and update cache
            for idx, embedding in zip(indices_to_process, new_embeddings):
                embeddings[idx] = embedding
                
                # Cache the embedding
                cache_key = cache_keys[idx] if cache_keys and idx < len(cache_keys) else texts[idx]
                self.embedding_cache[cache_key] = embedding
        
        # Save cache
        if self.config.cache_embeddings:
            self._save_cache()
        
        return embeddings
    
    async def _process_embedding_batches(self, texts: List[str]) -> List[List[float]]:
        """Process texts in batches to respect API limits."""
        all_embeddings = []
        
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i:i + self.config.batch_size]
            
            for attempt in range(self.config.max_retries):
                try:
                    batch_embeddings = await self._call_embedding_api(batch)
                    all_embeddings.extend(batch_embeddings)
                    break
                    
                except Exception as e:
                    logger.warning(f"Embedding API attempt {attempt + 1} failed: {e}")
                    if attempt == self.config.max_retries - 1:
                        # Use zero vectors as fallback
                        logger.error(f"Failed to get embeddings for batch, using zero vectors")
                        fallback_embeddings = [[0.0] * 1536] * len(batch)  # Ada-002 dimension
                        all_embeddings.extend(fallback_embeddings)
                    else:
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        return all_embeddings
    
    async def _call_embedding_api(self, texts: List[str]) -> List[List[float]]:
        """Make API call to generate embeddings."""
        response = await self.client.post("/embeddings", json={
            "model": self.config.model,
            "input": texts,
            "encoding_format": "float"
        })
        
        response.raise_for_status()
        result = response.json()
        
        if 'data' not in result:
            raise Exception(f"Unexpected API response: {result}")
        
        # Extract embeddings in order
        embeddings = [None] * len(texts)
        for item in result['data']:
            embeddings[item['index']] = item['embedding']
        
        return embeddings
    
    async def embed_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Add embeddings to entity objects.
        
        Args:
            entities: List of entity dictionaries
            
        Returns:
            Entities with embedding vectors added
        """
        if not entities:
            return entities
        
        # Prepare texts for embedding
        texts = []
        cache_keys = []
        
        for entity in entities:
            # Create comprehensive text representation
            text_parts = [
                entity.get('name', ''),
                entity.get('type', ''),
                entity.get('description', ''),
            ]
            
            # Add properties
            properties = entity.get('properties', {})
            if properties:
                text_parts.extend([f"{k}: {v}" for k, v in properties.items() if isinstance(v, str)])
            
            entity_text = " | ".join(filter(None, text_parts))
            texts.append(entity_text)
            cache_keys.append(f"entity_{entity.get('id', entity.get('name', ''))}")
        
        # Generate embeddings
        embeddings = await self.generate_embeddings(texts, cache_keys)
        
        # Add embeddings to entities
        embedded_entities = []
        for entity, embedding in zip(entities, embeddings):
            embedded_entity = entity.copy()
            embedded_entity['embedding'] = embedding
            embedded_entity['embedding_text'] = texts[len(embedded_entities)]
            embedded_entities.append(embedded_entity)
        
        logger.info(f"Added embeddings to {len(embedded_entities)} entities")
        return embedded_entities
    
    async def embed_relationships(self, relationships: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Add embeddings to relationship objects.
        
        Args:
            relationships: List of relationship dictionaries
            
        Returns:
            Relationships with embedding vectors added
        """
        if not relationships:
            return relationships
        
        # Prepare texts for embedding
        texts = []
        cache_keys = []
        
        for rel in relationships:
            # Create relationship text representation
            text_parts = [
                rel.get('source', ''),
                rel.get('type', ''),
                rel.get('target', ''),
                rel.get('description', ''),
                rel.get('evidence', '')
            ]
            
            rel_text = " | ".join(filter(None, text_parts))
            texts.append(rel_text)
            cache_keys.append(f"rel_{rel.get('id', len(cache_keys))}")
        
        # Generate embeddings
        embeddings = await self.generate_embeddings(texts, cache_keys)
        
        # Add embeddings to relationships
        embedded_relationships = []
        for rel, embedding in zip(relationships, embeddings):
            embedded_rel = rel.copy()
            embedded_rel['embedding'] = embedding
            embedded_rel['embedding_text'] = texts[len(embedded_relationships)]
            embedded_relationships.append(embedded_rel)
        
        logger.info(f"Added embeddings to {len(embedded_relationships)} relationships")
        return embedded_relationships
    
    def find_similar_entities(self, 
                             query_entity: Dict[str, Any],
                             entity_pool: List[Dict[str, Any]],
                             similarity_threshold: float = 0.8,
                             top_k: int = 10) -> List[Tuple[Dict[str, Any], float]]:
        """
        Find entities similar to a query entity using cosine similarity.
        
        Args:
            query_entity: Entity to find similarities for
            entity_pool: Pool of entities to search in
            similarity_threshold: Minimum similarity score
            top_k: Maximum number of results
            
        Returns:
            List of (entity, similarity_score) tuples
        """
        if 'embedding' not in query_entity:
            logger.warning("Query entity has no embedding")
            return []
        
        # Filter entities with embeddings
        entities_with_embeddings = [e for e in entity_pool if 'embedding' in e]
        if not entities_with_embeddings:
            return []
        
        query_embedding = np.array(query_entity['embedding']).reshape(1, -1)
        entity_embeddings = np.array([e['embedding'] for e in entities_with_embeddings])
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, entity_embeddings)[0]
        
        # Filter and sort results
        results = []
        for entity, similarity in zip(entities_with_embeddings, similarities):
            if similarity >= similarity_threshold and entity.get('id') != query_entity.get('id'):
                results.append((entity, float(similarity)))
        
        # Sort by similarity and return top_k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def cluster_entities(self, 
                        entities: List[Dict[str, Any]], 
                        eps: float = 0.3,
                        min_samples: int = 2) -> Dict[str, List[Dict[str, Any]]]:
        """
        Cluster entities based on embedding similarity using DBSCAN.
        
        Args:
            entities: List of entities with embeddings
            eps: DBSCAN epsilon parameter (distance threshold)
            min_samples: Minimum samples per cluster
            
        Returns:
            Dictionary mapping cluster labels to entity lists
        """
        entities_with_embeddings = [e for e in entities if 'embedding' in e]
        
        if len(entities_with_embeddings) < min_samples:
            return {"cluster_0": entities_with_embeddings}
        
        # Prepare embeddings matrix
        embeddings_matrix = np.array([e['embedding'] for e in entities_with_embeddings])
        
        # Perform clustering
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
        cluster_labels = dbscan.fit_predict(embeddings_matrix)
        
        # Group entities by cluster
        clusters = {}
        for entity, label in zip(entities_with_embeddings, cluster_labels):
            cluster_key = f"cluster_{label}" if label != -1 else "noise"
            if cluster_key not in clusters:
                clusters[cluster_key] = []
            clusters[cluster_key].append(entity)
        
        logger.info(f"Clustered {len(entities_with_embeddings)} entities into {len(clusters)} groups")
        return clusters
    
    def resolve_entity_duplicates(self, 
                                entities: List[Dict[str, Any]], 
                                similarity_threshold: float = 0.9) -> List[Dict[str, Any]]:
        """
        Resolve entity duplicates by merging similar entities.
        
        Args:
            entities: List of entities to deduplicate
            similarity_threshold: Threshold for considering entities as duplicates
            
        Returns:
            List of deduplicated entities
        """
        if not entities:
            return entities
        
        entities_with_embeddings = [e for e in entities if 'embedding' in e]
        entities_without_embeddings = [e for e in entities if 'embedding' not in e]
        
        if len(entities_with_embeddings) < 2:
            return entities
        
        # Build similarity matrix
        embeddings_matrix = np.array([e['embedding'] for e in entities_with_embeddings])
        similarity_matrix = cosine_similarity(embeddings_matrix)
        
        # Find duplicate pairs
        merged_indices = set()
        resolved_entities = []
        
        for i, entity in enumerate(entities_with_embeddings):
            if i in merged_indices:
                continue
            
            # Find similar entities
            duplicates = [i]
            for j in range(i + 1, len(entities_with_embeddings)):
                if j not in merged_indices and similarity_matrix[i][j] >= similarity_threshold:
                    duplicates.append(j)
                    merged_indices.add(j)
            
            # Merge duplicates
            if len(duplicates) > 1:
                merged_entity = self._merge_entities([entities_with_embeddings[idx] for idx in duplicates])
                resolved_entities.append(merged_entity)
                logger.info(f"Merged {len(duplicates)} duplicate entities: {merged_entity['name']}")
            else:
                resolved_entities.append(entity)
        
        # Add entities without embeddings
        resolved_entities.extend(entities_without_embeddings)
        
        logger.info(f"Entity resolution: {len(entities)} → {len(resolved_entities)} entities")
        return resolved_entities
    
    def _merge_entities(self, entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge multiple entities into a single entity."""
        if not entities:
            return {}
        
        if len(entities) == 1:
            return entities[0]
        
        # Start with the first entity
        merged = entities[0].copy()
        
        # Merge names (prefer longer/more descriptive)
        names = [e.get('name', '') for e in entities]
        merged['name'] = max(names, key=len)
        
        # Merge types (prefer most common or first non-empty)
        types = [e.get('type', '') for e in entities if e.get('type')]
        if types:
            merged['type'] = max(set(types), key=types.count)
        
        # Merge descriptions (combine non-empty ones)
        descriptions = [e.get('description', '') for e in entities if e.get('description')]
        if descriptions:
            merged['description'] = ' | '.join(set(descriptions))
        
        # Average confidence scores
        confidences = [e.get('confidence', 0.0) for e in entities]
        merged['confidence'] = sum(confidences) / len(confidences)
        
        # Merge properties
        all_properties = {}
        for entity in entities:
            props = entity.get('properties', {})
            all_properties.update(props)
        merged['properties'] = all_properties
        
        # Keep the best embedding (highest confidence)
        best_entity = max(entities, key=lambda e: e.get('confidence', 0.0))
        if 'embedding' in best_entity:
            merged['embedding'] = best_entity['embedding']
        
        # Track merge metadata
        merged['merged_from'] = [e.get('id', e.get('name', '')) for e in entities]
        merged['merge_timestamp'] = datetime.utcnow().isoformat()
        
        return merged
    
    def _load_cache(self):
        """Load embedding cache from file."""
        try:
            with open(self.cache_file, 'rb') as f:
                self.embedding_cache = pickle.load(f)
            logger.info(f"Loaded {len(self.embedding_cache)} cached embeddings")
        except Exception as e:
            logger.warning(f"Failed to load embedding cache: {e}")
            self.embedding_cache = {}
    
    def _save_cache(self):
        """Save embedding cache to file."""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.embedding_cache, f)
            logger.debug(f"Saved {len(self.embedding_cache)} embeddings to cache")
        except Exception as e:
            logger.warning(f"Failed to save embedding cache: {e}")
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get embedding cache statistics."""
        return {
            'cache_size': len(self.embedding_cache),
            'cache_file': self.cache_file,
            'model': self.config.model,
            'total_dimensions': len(next(iter(self.embedding_cache.values()), [])),
        }
    
    async def close(self):
        """Close HTTP client and save cache."""
        if self.config.cache_embeddings:
            self._save_cache()
        await self.client.aclose()


# Utility functions
def load_embedding_config_from_env() -> EmbeddingConfig:
    """Load embedding configuration from environment variables."""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    return EmbeddingConfig(
        api_key=api_key,
        base_url=os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1'),
        model=os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-ada-002'),
        batch_size=int(os.getenv('EMBEDDING_BATCH_SIZE', '100')),
        cache_embeddings=os.getenv('CACHE_EMBEDDINGS', 'true').lower() == 'true'
    )


async def demo_embeddings():
    """Demonstration of embedding functionality."""
    # Sample entities
    entities = [
        {"id": "1", "name": "John Smith", "type": "person", "description": "Software Engineer"},
        {"id": "2", "name": "John Smith", "type": "person", "description": "Senior Developer"},  # Duplicate
        {"id": "3", "name": "Microsoft", "type": "organization", "description": "Technology company"},
        {"id": "4", "name": "Azure AI", "type": "technology", "description": "Cloud AI platform"},
    ]
    
    try:
        config = load_embedding_config_from_env()
        manager = EmbeddingManager(config)
        
        print("Generating embeddings for entities...")
        embedded_entities = await manager.embed_entities(entities)
        
        print(f"Generated embeddings for {len(embedded_entities)} entities")
        
        # Test similarity search
        print("\nTesting similarity search...")
        similar_entities = manager.find_similar_entities(
            embedded_entities[0], 
            embedded_entities[1:],
            similarity_threshold=0.7
        )
        
        for entity, similarity in similar_entities:
            print(f"Similar: {entity['name']} (similarity: {similarity:.3f})")
        
        # Test entity resolution
        print("\nTesting entity resolution...")
        resolved_entities = manager.resolve_entity_duplicates(embedded_entities, similarity_threshold=0.9)
        print(f"Resolved: {len(entities)} → {len(resolved_entities)} entities")
        
        # Cache statistics
        cache_stats = manager.get_cache_statistics()
        print(f"\nCache: {cache_stats['cache_size']} embeddings stored")
        
        await manager.close()
        
    except Exception as e:
        print(f"Demo failed: {e}")
        print("Make sure to set OPENAI_API_KEY environment variable")


if __name__ == "__main__":
    asyncio.run(demo_embeddings())
#!/usr/bin/env python3
"""
Cypher Traversal Engine for Agentic Graph RAG
Advanced Neo4j Cypher query engine for graph traversal and pattern matching.
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional, Union, Tuple, Set
from dataclasses import dataclass
from datetime import datetime
import json
import re

try:
    from neo4j import GraphDatabase, Driver
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    logging.warning("Neo4j driver not available. Install with: pip install neo4j")

logger = logging.getLogger(__name__)


@dataclass
class CypherQuery:
    """Represents a Cypher query with metadata."""
    query: str
    parameters: Dict[str, Any] = None
    description: str = ""
    query_type: str = "read"  # read, write, aggregate, traversal
    complexity: int = 1  # 1-5 complexity score
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}


@dataclass
class GraphPattern:
    """Graph pattern for template-based query generation."""
    name: str
    pattern: str
    parameters: List[str]
    description: str
    category: str = "general"


@dataclass
class TraversalResult:
    """Result from graph traversal operation."""
    nodes: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    paths: List[List[Dict[str, Any]]]
    statistics: Dict[str, Any]
    query: str
    execution_time: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'nodes': self.nodes,
            'relationships': self.relationships,
            'paths': self.paths,
            'statistics': self.statistics,
            'query': self.query,
            'execution_time': self.execution_time
        }


class CypherTraversalEngine:
    """
    Advanced Cypher query engine for knowledge graph operations.
    
    Features:
    - Template-based query generation
    - Pattern matching and traversal
    - Query optimization and caching
    - Relationship inference
    - Multi-hop path finding
    - Graph analytics and metrics
    """
    
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str):
        """Initialize Cypher traversal engine."""
        if not NEO4J_AVAILABLE:
            raise ImportError("Neo4j driver is required. Install with: pip install neo4j")
        
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.driver: Optional[Driver] = None
        
        # Query cache
        self.query_cache: Dict[str, Any] = {}
        self.cache_enabled = True
        
        # Built-in graph patterns
        self.graph_patterns = self._initialize_patterns()
        
        # Connection tracking
        self.connected = False
        
        logger.info("Cypher Traversal Engine initialized")
    
    def _initialize_patterns(self) -> Dict[str, GraphPattern]:
        """Initialize built-in graph patterns."""
        patterns = {}
        
        # Entity-centric patterns
        patterns['entity_neighbors'] = GraphPattern(
            name="entity_neighbors",
            pattern="MATCH (e:Entity {name: $entity_name})-[r]-(n) RETURN e, r, n",
            parameters=["entity_name"],
            description="Find all neighbors of an entity",
            category="entity"
        )
        
        patterns['entity_relationships'] = GraphPattern(
            name="entity_relationships", 
            pattern="MATCH (e1:Entity {name: $entity1})-[r]-(e2:Entity {name: $entity2}) RETURN e1, r, e2",
            parameters=["entity1", "entity2"],
            description="Find relationships between two entities",
            category="entity"
        )
        
        # Path finding patterns
        patterns['shortest_path'] = GraphPattern(
            name="shortest_path",
            pattern="MATCH path = shortestPath((e1:Entity {name: $start})-[*..10]-(e2:Entity {name: $end})) RETURN path",
            parameters=["start", "end"],
            description="Find shortest path between entities",
            category="path"
        )
        
        patterns['all_paths'] = GraphPattern(
            name="all_paths",
            pattern="MATCH path = (e1:Entity {name: $start})-[*1..$max_depth]-(e2:Entity {name: $end}) RETURN path LIMIT $limit",
            parameters=["start", "end", "max_depth", "limit"],
            description="Find all paths between entities",
            category="path"
        )
        
        # Semantic patterns
        patterns['similar_entities'] = GraphPattern(
            name="similar_entities",
            pattern="MATCH (e:Entity) WHERE e.embedding IS NOT NULL AND gds.similarity.cosine(e.embedding, $query_embedding) > $threshold RETURN e ORDER BY gds.similarity.cosine(e.embedding, $query_embedding) DESC LIMIT $limit",
            parameters=["query_embedding", "threshold", "limit"],
            description="Find entities similar to query embedding",
            category="semantic"
        )
        
        # Analytical patterns
        patterns['entity_centrality'] = GraphPattern(
            name="entity_centrality",
            pattern="MATCH (e:Entity)-[r]-() RETURN e.name as entity, count(r) as degree ORDER BY degree DESC LIMIT $limit",
            parameters=["limit"],
            description="Find most connected entities",
            category="analytics"
        )
        
        patterns['relationship_types'] = GraphPattern(
            name="relationship_types",
            pattern="MATCH ()-[r]->() RETURN type(r) as relationship_type, count(r) as count ORDER BY count DESC",
            parameters=[],
            description="Get relationship type distribution",
            category="analytics"
        )
        
        # Context patterns
        patterns['entity_context'] = GraphPattern(
            name="entity_context",
            pattern="MATCH (e:Entity {name: $entity_name})-[r1]-(n1)-[r2]-(n2) WHERE n2 <> e RETURN e, r1, n1, r2, n2 LIMIT $limit",
            parameters=["entity_name", "limit"],
            description="Find entity's extended context (2-hop neighbors)",
            category="context"
        )
        
        return patterns
    
    async def connect(self) -> bool:
        """Connect to Neo4j database."""
        try:
            self.driver = GraphDatabase.driver(
                self.neo4j_uri,
                auth=(self.neo4j_user, self.neo4j_password)
            )
            
            # Test connection
            async with self.driver.session() as session:
                result = await session.run("RETURN 1 as test")
                await result.consume()
            
            self.connected = True
            logger.info("Connected to Neo4j database")
            return True
        
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        """Disconnect from Neo4j database."""
        if self.driver:
            self.driver.close()
            self.connected = False
            logger.info("Disconnected from Neo4j database")
    
    async def execute_cypher(self, 
                           query: str, 
                           parameters: Dict[str, Any] = None,
                           use_cache: bool = True) -> TraversalResult:
        """
        Execute Cypher query and return structured results.
        
        Args:
            query: Cypher query string
            parameters: Query parameters
            use_cache: Whether to use query cache
            
        Returns:
            TraversalResult with nodes, relationships, and metadata
        """
        if not self.connected:
            await self.connect()
        
        if parameters is None:
            parameters = {}
        
        # Check cache
        cache_key = f"{query}:{json.dumps(parameters, sort_keys=True)}"
        if use_cache and self.cache_enabled and cache_key in self.query_cache:
            logger.debug("Returning cached query result")
            return self.query_cache[cache_key]
        
        start_time = datetime.now()
        
        try:
            async with self.driver.session() as session:
                result = await session.run(query, parameters)
                records = await result.data()
                summary = await result.consume()
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Process results
            nodes = []
            relationships = []
            paths = []
            
            for record in records:
                for key, value in record.items():
                    if hasattr(value, 'labels'):  # Node
                        node_data = dict(value)
                        node_data['labels'] = list(value.labels)
                        node_data['id'] = value.id
                        nodes.append(node_data)
                    
                    elif hasattr(value, 'type'):  # Relationship
                        rel_data = dict(value)
                        rel_data['type'] = value.type
                        rel_data['id'] = value.id
                        rel_data['start_node'] = value.start_node.id
                        rel_data['end_node'] = value.end_node.id
                        relationships.append(rel_data)
                    
                    elif hasattr(value, 'nodes') and hasattr(value, 'relationships'):  # Path
                        path_data = {
                            'nodes': [dict(n) for n in value.nodes],
                            'relationships': [dict(r) for r in value.relationships],
                            'length': len(value.relationships)
                        }
                        paths.append(path_data)
            
            # Create result
            traversal_result = TraversalResult(
                nodes=self._deduplicate_nodes(nodes),
                relationships=self._deduplicate_relationships(relationships),
                paths=paths,
                statistics={
                    'records_returned': len(records),
                    'execution_time': execution_time,
                    'query_type': self._classify_query(query),
                    'nodes_accessed': summary.counters.nodes_created + summary.counters.nodes_deleted,
                    'relationships_accessed': summary.counters.relationships_created + summary.counters.relationships_deleted
                },
                query=query,
                execution_time=execution_time
            )
            
            # Cache result
            if use_cache and self.cache_enabled:
                self.query_cache[cache_key] = traversal_result
            
            logger.debug(f"Query executed in {execution_time:.3f}s, returned {len(records)} records")
            return traversal_result
        
        except Exception as e:
            logger.error(f"Cypher query failed: {e}")
            return TraversalResult(
                nodes=[], relationships=[], paths=[],
                statistics={'error': str(e)},
                query=query,
                execution_time=(datetime.now() - start_time).total_seconds()
            )
    
    def _deduplicate_nodes(self, nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate nodes based on ID."""
        seen_ids = set()
        unique_nodes = []
        
        for node in nodes:
            node_id = node.get('id')
            if node_id not in seen_ids:
                seen_ids.add(node_id)
                unique_nodes.append(node)
        
        return unique_nodes
    
    def _deduplicate_relationships(self, relationships: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate relationships based on ID."""
        seen_ids = set()
        unique_rels = []
        
        for rel in relationships:
            rel_id = rel.get('id')
            if rel_id not in seen_ids:
                seen_ids.add(rel_id)
                unique_rels.append(rel)
        
        return unique_rels
    
    def _classify_query(self, query: str) -> str:
        """Classify query type based on keywords."""
        query_upper = query.upper()
        
        if 'CREATE' in query_upper or 'MERGE' in query_upper:
            return 'write'
        elif 'DELETE' in query_upper or 'REMOVE' in query_upper:
            return 'delete'
        elif 'COUNT' in query_upper or 'SUM' in query_upper or 'AVG' in query_upper:
            return 'aggregate'
        elif 'MATCH' in query_upper and ('SHORTESTPATH' in query_upper or '*' in query):
            return 'traversal'
        else:
            return 'read'
    
    async def find_entity_neighbors(self, entity_name: str, max_depth: int = 1) -> TraversalResult:
        """Find all neighbors of an entity within specified depth."""
        if max_depth == 1:
            pattern = self.graph_patterns['entity_neighbors']
            return await self.execute_cypher(pattern.pattern, {'entity_name': entity_name})
        else:
            query = f"MATCH (e:Entity {{name: $entity_name}})-[r*1..{max_depth}]-(n) RETURN e, r, n"
            return await self.execute_cypher(query, {'entity_name': entity_name})
    
    async def find_shortest_path(self, start_entity: str, end_entity: str, max_length: int = 10) -> TraversalResult:
        """Find shortest path between two entities."""
        query = f"MATCH path = shortestPath((e1:Entity {{name: $start}})-[*..{max_length}]-(e2:Entity {{name: $end}})) RETURN path"
        return await self.execute_cypher(query, {'start': start_entity, 'end': end_entity})
    
    async def find_all_paths(self, start_entity: str, end_entity: str, max_depth: int = 5, limit: int = 10) -> TraversalResult:
        """Find all paths between entities within depth limit."""
        pattern = self.graph_patterns['all_paths']
        parameters = {
            'start': start_entity,
            'end': end_entity,
            'max_depth': max_depth,
            'limit': limit
        }
        return await self.execute_cypher(pattern.pattern, parameters)
    
    async def find_similar_entities(self, query_embedding: List[float], threshold: float = 0.7, limit: int = 10) -> TraversalResult:
        """Find entities similar to query embedding."""
        # Note: This requires Neo4j with GDS library
        query = """
        MATCH (e:Entity) 
        WHERE e.embedding IS NOT NULL 
        WITH e, gds.similarity.cosine(e.embedding, $query_embedding) as similarity
        WHERE similarity > $threshold
        RETURN e, similarity
        ORDER BY similarity DESC 
        LIMIT $limit
        """
        parameters = {
            'query_embedding': query_embedding,
            'threshold': threshold,
            'limit': limit
        }
        return await self.execute_cypher(query, parameters)
    
    async def get_entity_context(self, entity_name: str, context_depth: int = 2, limit: int = 50) -> TraversalResult:
        """Get extended context around an entity."""
        if context_depth == 1:
            return await self.find_entity_neighbors(entity_name)
        
        query = f"""
        MATCH path = (e:Entity {{name: $entity_name}})-[*1..{context_depth}]-(n)
        RETURN path
        LIMIT $limit
        """
        return await self.execute_cypher(query, {'entity_name': entity_name, 'limit': limit})
    
    async def analyze_graph_structure(self) -> Dict[str, Any]:
        """Analyze overall graph structure and provide statistics."""
        queries = {
            'node_count': "MATCH (n) RETURN count(n) as count",
            'relationship_count': "MATCH ()-[r]->() RETURN count(r) as count", 
            'entity_count': "MATCH (n:Entity) RETURN count(n) as count",
            'relationship_types': "MATCH ()-[r]->() RETURN type(r) as type, count(r) as count ORDER BY count DESC",
            'most_connected': "MATCH (e:Entity)-[r]-() RETURN e.name as entity, count(r) as degree ORDER BY degree DESC LIMIT 10"
        }
        
        analysis = {}
        
        for name, query in queries.items():
            try:
                result = await self.execute_cypher(query)
                if result.nodes:
                    analysis[name] = result.nodes
                else:
                    # Handle aggregate queries
                    async with self.driver.session() as session:
                        result = await session.run(query)
                        records = await result.data()
                        analysis[name] = records
            except Exception as e:
                logger.error(f"Failed to execute analysis query {name}: {e}")
                analysis[name] = []
        
        return analysis
    
    async def find_relationship_patterns(self, entity_name: str, pattern_depth: int = 2) -> List[Dict[str, Any]]:
        """Find common relationship patterns around an entity."""
        query = f"""
        MATCH path = (e:Entity {{name: $entity_name}})-[r1*1..{pattern_depth}]->(n)
        WITH [rel in relationships(path) | type(rel)] as pattern, count(*) as frequency
        WHERE frequency > 1
        RETURN pattern, frequency
        ORDER BY frequency DESC
        LIMIT 20
        """
        
        result = await self.execute_cypher(query, {'entity_name': entity_name})
        
        # Process into pattern format
        patterns = []
        async with self.driver.session() as session:
            result = await session.run(query, {'entity_name': entity_name})
            records = await result.data()
            
            for record in records:
                patterns.append({
                    'pattern': ' -> '.join(record['pattern']),
                    'frequency': record['frequency']
                })
        
        return patterns
    
    def clear_cache(self):
        """Clear query cache."""
        self.query_cache.clear()
        logger.info("Query cache cleared")
    
    def add_custom_pattern(self, pattern: GraphPattern):
        """Add custom graph pattern."""
        self.graph_patterns[pattern.name] = pattern
        logger.info(f"Added custom pattern: {pattern.name}")
    
    def get_available_patterns(self) -> List[str]:
        """Get list of available graph patterns."""
        return list(self.graph_patterns.keys())
    
    async def close(self):
        """Clean up resources."""
        self.disconnect()
        self.clear_cache()
        logger.info("Cypher Traversal Engine closed")


# Convenience functions
async def create_cypher_engine(neo4j_uri: str, 
                             neo4j_user: str, 
                             neo4j_password: str) -> CypherTraversalEngine:
    """Create and connect Cypher traversal engine."""
    engine = CypherTraversalEngine(neo4j_uri, neo4j_user, neo4j_password)
    await engine.connect()
    return engine


async def execute_graph_query(engine: CypherTraversalEngine,
                            query_text: str,
                            query_type: str = "general") -> TraversalResult:
    """Execute a graph query based on natural language description."""
    # This would benefit from NL-to-Cypher translation
    # For now, we'll use pattern matching
    
    query_lower = query_text.lower()
    
    if "neighbors" in query_lower or "connected to" in query_lower:
        # Extract entity name (simple pattern matching)
        entity_match = re.search(r"of (.+?)(\s|$)", query_lower)
        if entity_match:
            entity_name = entity_match.group(1).strip()
            return await engine.find_entity_neighbors(entity_name)
    
    elif "path" in query_lower and "between" in query_lower:
        # Extract entity names for path finding
        entities = re.findall(r"between (.+?) and (.+?)(\s|$)", query_lower)
        if entities:
            start, end = entities[0][:2]
            return await engine.find_shortest_path(start.strip(), end.strip())
    
    # Default: return empty result with suggestion
    return TraversalResult(
        nodes=[], relationships=[], paths=[],
        statistics={'suggestion': 'Try queries like "neighbors of EntityName" or "path between Entity1 and Entity2"'},
        query=query_text,
        execution_time=0.0
    )
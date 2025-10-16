"""
Graph traversal agent with Cypher-based relationship querying and path reasoning.

This module provides sophisticated graph traversal capabilities, including
relationship-based queries, path finding, and multi-step reasoning chains
using the Neo4j Cypher query language.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, asdict
from enum import Enum
import json

from core.graph_store import GraphStore, GraphNode, GraphRelationship, GraphQuery, QueryResult, RelationType

logger = logging.getLogger(__name__)


class TraversalDirection(Enum):
    """Direction for graph traversal."""
    OUTGOING = "outgoing"
    INCOMING = "incoming" 
    BOTH = "both"


class PathPattern(Enum):
    """Common path patterns for graph queries."""
    SHORTEST_PATH = "shortest_path"
    ALL_PATHS = "all_paths"
    DIRECT_NEIGHBORS = "direct_neighbors"
    MULTI_HOP = "multi_hop"
    SEMANTIC_PATH = "semantic_path"


@dataclass
class TraversalQuery:
    """Query for graph traversal operations."""
    start_node_id: Optional[str] = None
    end_node_id: Optional[str] = None
    start_node_name: Optional[str] = None
    end_node_name: Optional[str] = None
    relationship_types: Optional[List[str]] = None
    direction: TraversalDirection = TraversalDirection.BOTH
    max_depth: int = 3
    max_results: int = 100
    pattern: PathPattern = PathPattern.DIRECT_NEIGHBORS
    property_filters: Optional[Dict[str, Any]] = None
    exclude_node_ids: Optional[List[str]] = None


@dataclass
class GraphPath:
    """Represents a path through the graph."""
    nodes: List[GraphNode]
    relationships: List[GraphRelationship]
    path_length: int
    total_confidence: float
    path_reasoning: str
    metadata: Dict[str, Any]


@dataclass
class TraversalResult:
    """Result of a graph traversal operation."""
    paths: List[GraphPath]
    related_nodes: List[GraphNode]
    query_reasoning: str
    execution_time: float
    cypher_queries: List[str]
    total_nodes_visited: int


@dataclass
class EntityNeighborhood:
    """Represents the local neighborhood around an entity."""
    central_node: GraphNode
    direct_neighbors: List[GraphNode]
    relationships: List[GraphRelationship] 
    neighbor_types: Dict[str, int]
    relationship_types: Dict[str, int]
    depth_analysis: Dict[int, int]  # depth -> node_count


class GraphTraversalAgent:
    """
    Provides sophisticated graph traversal and relationship analysis capabilities.
    
    This agent can execute complex graph queries, find paths between entities,
    analyze neighborhoods, and provide reasoning about relationship patterns.
    """
    
    def __init__(self, 
                 graph_store: GraphStore,
                 default_max_depth: int = 3,
                 default_max_results: int = 100,
                 confidence_threshold: float = 0.5):
        """
        Initialize the graph traversal agent.
        
        Args:
            graph_store: Graph storage backend
            default_max_depth: Default maximum traversal depth
            default_max_results: Default maximum number of results
            confidence_threshold: Minimum confidence for including relationships
        """
        self.graph_store = graph_store
        self.default_max_depth = default_max_depth
        self.default_max_results = default_max_results
        self.confidence_threshold = confidence_threshold
    
    async def relationship_search(self, query: TraversalQuery) -> TraversalResult:
        """
        Perform relationship-based search in the graph.
        
        Args:
            query: Traversal query parameters
            
        Returns:
            TraversalResult: Results of the traversal operation
        """
        start_time = None
        try:
            import time
            start_time = time.time()
            
            # Resolve node IDs if only names provided
            start_node_id = await self._resolve_node_id(query.start_node_id, query.start_node_name)
            end_node_id = await self._resolve_node_id(query.end_node_id, query.end_node_name) if query.end_node_name else None
            
            if not start_node_id:
                return TraversalResult([], [], "No valid start node found", 0.0, [], 0)
            
            # Choose appropriate query pattern
            if query.pattern == PathPattern.SHORTEST_PATH and end_node_id:
                result = await self._find_shortest_path(start_node_id, end_node_id, query)
            elif query.pattern == PathPattern.ALL_PATHS and end_node_id:
                result = await self._find_all_paths(start_node_id, end_node_id, query)
            elif query.pattern == PathPattern.DIRECT_NEIGHBORS:
                result = await self._get_direct_neighbors(start_node_id, query)
            elif query.pattern == PathPattern.MULTI_HOP:
                result = await self._multi_hop_traversal(start_node_id, query)
            elif query.pattern == PathPattern.SEMANTIC_PATH:
                result = await self._semantic_path_search(start_node_id, query)
            else:
                result = await self._get_direct_neighbors(start_node_id, query)
            
            execution_time = time.time() - start_time if start_time else 0.0
            result.execution_time = execution_time
            
            logger.debug(f"Graph traversal completed in {execution_time:.3f}s: "
                        f"{len(result.paths)} paths, {len(result.related_nodes)} nodes")
            
            return result
            
        except Exception as e:
            logger.error(f"Graph traversal failed: {e}")
            execution_time = time.time() - start_time if start_time else 0.0
            return TraversalResult([], [], f"Traversal failed: {e}", execution_time, [], 0)
    
    async def _resolve_node_id(self, node_id: Optional[str], node_name: Optional[str]) -> Optional[str]:
        """Resolve node ID from either ID or name."""
        if node_id:
            return node_id
        
        if node_name:
            # Search for node by name
            nodes = await self.graph_store.find_nodes_by_property("name", node_name)
            if nodes:
                return nodes[0].id
        
        return None
    
    async def _find_shortest_path(self, start_id: str, end_id: str, query: TraversalQuery) -> TraversalResult:
        """Find the shortest path between two nodes."""
        rel_filter = self._build_relationship_filter(query.relationship_types)
        
        cypher = f"""
        MATCH (start) WHERE elementId(start) = $start_id
        MATCH (end) WHERE elementId(end) = $end_id
        MATCH path = shortestPath((start)-[{rel_filter}*1..{query.max_depth}]-(end))
        WHERE ALL(rel in relationships(path) WHERE rel.confidence >= $confidence)
        RETURN path, length(path) as path_length
        LIMIT $limit
        """
        
        cypher_query = GraphQuery(
            cypher=cypher,
            parameters={
                "start_id": start_id,
                "end_id": end_id,
                "confidence": self.confidence_threshold,
                "limit": query.max_results
            }
        )
        
        result = await self.graph_store.query(cypher_query)
        paths = await self._parse_path_results(result)
        
        reasoning = f"Found shortest path between nodes using max depth {query.max_depth}"
        
        return TraversalResult(
            paths=paths,
            related_nodes=[],
            query_reasoning=reasoning,
            execution_time=0.0,
            cypher_queries=[cypher],
            total_nodes_visited=len(set(node.id for path in paths for node in path.nodes))
        )
    
    async def _find_all_paths(self, start_id: str, end_id: str, query: TraversalQuery) -> TraversalResult:
        """Find all paths between two nodes within depth limit."""
        rel_filter = self._build_relationship_filter(query.relationship_types)
        
        cypher = f"""
        MATCH (start) WHERE elementId(start) = $start_id
        MATCH (end) WHERE elementId(end) = $end_id
        MATCH path = (start)-[{rel_filter}*1..{query.max_depth}]-(end)
        WHERE ALL(rel in relationships(path) WHERE rel.confidence >= $confidence)
        AND length(path) <= $max_depth
        RETURN path, length(path) as path_length
        ORDER BY path_length
        LIMIT $limit
        """
        
        cypher_query = GraphQuery(
            cypher=cypher,
            parameters={
                "start_id": start_id,
                "end_id": end_id,
                "confidence": self.confidence_threshold,
                "max_depth": query.max_depth,
                "limit": query.max_results
            }
        )
        
        result = await self.graph_store.query(cypher_query)
        paths = await self._parse_path_results(result)
        
        reasoning = f"Found {len(paths)} paths between nodes within depth {query.max_depth}"
        
        return TraversalResult(
            paths=paths,
            related_nodes=[],
            query_reasoning=reasoning,
            execution_time=0.0,
            cypher_queries=[cypher],
            total_nodes_visited=len(set(node.id for path in paths for node in path.nodes))
        )
    
    async def _get_direct_neighbors(self, start_id: str, query: TraversalQuery) -> TraversalResult:
        """Get direct neighbors of a node."""
        direction_pattern = self._build_direction_pattern(query.direction)
        rel_filter = self._build_relationship_filter(query.relationship_types)
        
        cypher = f"""
        MATCH (start) WHERE elementId(start) = $start_id
        MATCH (start){direction_pattern}[r{rel_filter}](neighbor)
        WHERE r.confidence >= $confidence
        {self._build_property_filter_clause(query.property_filters, "neighbor")}
        RETURN neighbor, r, elementId(neighbor) as neighbor_id, elementId(r) as rel_id
        LIMIT $limit
        """
        
        cypher_query = GraphQuery(
            cypher=cypher,
            parameters={
                "start_id": start_id,
                "confidence": self.confidence_threshold,
                "limit": query.max_results
            }
        )
        
        result = await self.graph_store.query(cypher_query)
        
        # Build simple paths (length 1)
        paths = []
        related_nodes = []
        
        start_node = await self.graph_store.get_node(start_id)
        
        for record in result.nodes:
            if record.id != start_id:  # Skip the start node itself
                related_nodes.append(record)
                
                # Create simple path
                path = GraphPath(
                    nodes=[start_node, record] if start_node else [record],
                    relationships=[],  # Will be filled from relationships in result
                    path_length=1,
                    total_confidence=1.0,
                    path_reasoning="Direct neighbor",
                    metadata={"pattern": "direct_neighbors"}
                )
                paths.append(path)
        
        reasoning = f"Found {len(related_nodes)} direct neighbors"
        
        return TraversalResult(
            paths=paths,
            related_nodes=related_nodes,
            query_reasoning=reasoning,
            execution_time=0.0,
            cypher_queries=[cypher],
            total_nodes_visited=len(related_nodes) + 1
        )
    
    async def _multi_hop_traversal(self, start_id: str, query: TraversalQuery) -> TraversalResult:
        """Perform multi-hop traversal to explore the neighborhood."""
        rel_filter = self._build_relationship_filter(query.relationship_types)
        
        cypher = f"""
        MATCH (start) WHERE elementId(start) = $start_id
        MATCH path = (start)-[{rel_filter}*1..{query.max_depth}]-(node)
        WHERE ALL(rel in relationships(path) WHERE rel.confidence >= $confidence)
        {self._build_property_filter_clause(query.property_filters, "node")}
        RETURN path, node, length(path) as depth
        ORDER BY depth, node.name
        LIMIT $limit
        """
        
        cypher_query = GraphQuery(
            cypher=cypher,
            parameters={
                "start_id": start_id,
                "confidence": self.confidence_threshold,
                "limit": query.max_results
            }
        )
        
        result = await self.graph_store.query(cypher_query)
        paths = await self._parse_path_results(result)
        
        # Extract unique nodes
        related_nodes = []
        seen_ids = set()
        
        for path in paths:
            for node in path.nodes:
                if node.id not in seen_ids and node.id != start_id:
                    related_nodes.append(node)
                    seen_ids.add(node.id)
        
        reasoning = f"Explored {query.max_depth}-hop neighborhood, found {len(paths)} paths to {len(related_nodes)} nodes"
        
        return TraversalResult(
            paths=paths,
            related_nodes=related_nodes,
            query_reasoning=reasoning,
            execution_time=0.0,
            cypher_queries=[cypher],
            total_nodes_visited=len(seen_ids)
        )
    
    async def _semantic_path_search(self, start_id: str, query: TraversalQuery) -> TraversalResult:
        """Find semantically meaningful paths."""
        # This is a more advanced query that considers relationship semantics
        cypher = f"""
        MATCH (start) WHERE elementId(start) = $start_id
        MATCH path = (start)-[r1:IS_A|PART_OF|HAS_PROPERTY*1..{query.max_depth}]-(node)
        WHERE ALL(rel in relationships(path) WHERE rel.confidence >= $confidence)
        RETURN path, 
               reduce(conf = 1.0, rel in relationships(path) | conf * rel.confidence) as path_confidence
        ORDER BY path_confidence DESC, length(path)
        LIMIT $limit
        """
        
        cypher_query = GraphQuery(
            cypher=cypher,
            parameters={
                "start_id": start_id,
                "confidence": self.confidence_threshold,
                "limit": query.max_results
            }
        )
        
        result = await self.graph_store.query(cypher_query)
        paths = await self._parse_path_results(result)
        
        reasoning = f"Found {len(paths)} semantically meaningful paths using hierarchical relationships"
        
        return TraversalResult(
            paths=paths,
            related_nodes=[],
            query_reasoning=reasoning,
            execution_time=0.0,
            cypher_queries=[cypher],
            total_nodes_visited=len(set(node.id for path in paths for node in path.nodes))
        )
    
    def _build_relationship_filter(self, relationship_types: Optional[List[str]]) -> str:
        """Build Cypher relationship type filter."""
        if not relationship_types:
            return ""
        
        # Clean and validate relationship types
        valid_types = []
        for rel_type in relationship_types:
            cleaned = rel_type.upper().replace(" ", "_")
            valid_types.append(cleaned)
        
        return ":" + "|".join(valid_types)
    
    def _build_direction_pattern(self, direction: TraversalDirection) -> str:
        """Build Cypher direction pattern."""
        if direction == TraversalDirection.OUTGOING:
            return "-"
        elif direction == TraversalDirection.INCOMING:
            return "<-"
        else:  # BOTH
            return "-"
    
    def _build_property_filter_clause(self, property_filters: Optional[Dict[str, Any]], node_var: str) -> str:
        """Build WHERE clause for property filters."""
        if not property_filters:
            return ""
        
        conditions = []
        for key, value in property_filters.items():
            if isinstance(value, str):
                conditions.append(f"{node_var}.{key} = '{value}'")
            elif isinstance(value, (int, float)):
                conditions.append(f"{node_var}.{key} = {value}")
            elif isinstance(value, list):
                value_list = "', '".join(str(v) for v in value)
                conditions.append(f"{node_var}.{key} IN ['{value_list}']")
        
        if conditions:
            return "AND " + " AND ".join(conditions)
        return ""
    
    async def _parse_path_results(self, query_result: QueryResult) -> List[GraphPath]:
        """Parse Cypher query results into GraphPath objects."""
        # This is a simplified parser - in a real implementation,
        # you'd need to properly parse Neo4j path objects from the result
        paths = []
        
        # For now, create simple paths from the nodes and relationships in results
        if query_result.nodes:
            # Group nodes into paths based on result structure
            # This is a simplified version - actual implementation would be more complex
            
            for i in range(0, len(query_result.nodes), 2):  # Simplified grouping
                nodes_in_path = query_result.nodes[i:i+2] if i+1 < len(query_result.nodes) else [query_result.nodes[i]]
                rels_in_path = query_result.relationships[i:i+1] if i < len(query_result.relationships) else []
                
                if nodes_in_path:
                    # Calculate path confidence
                    confidence = 1.0
                    if rels_in_path:
                        confidence = sum(r.confidence for r in rels_in_path) / len(rels_in_path)
                    
                    # Generate reasoning
                    reasoning = self._generate_path_reasoning(nodes_in_path, rels_in_path)
                    
                    path = GraphPath(
                        nodes=nodes_in_path,
                        relationships=rels_in_path,
                        path_length=len(rels_in_path),
                        total_confidence=confidence,
                        path_reasoning=reasoning,
                        metadata={"source": "cypher_query"}
                    )
                    paths.append(path)
        
        return paths
    
    def _generate_path_reasoning(self, nodes: List[GraphNode], relationships: List[GraphRelationship]) -> str:
        """Generate human-readable reasoning for a path."""
        if len(nodes) == 1:
            return f"Single node: {nodes[0].name}"
        
        if len(nodes) == 2 and len(relationships) == 1:
            rel = relationships[0]
            return f"{nodes[0].name} {rel.type.value.lower().replace('_', ' ')} {nodes[1].name}"
        
        # Multi-hop path
        path_desc = []
        for i, node in enumerate(nodes):
            path_desc.append(node.name)
            if i < len(relationships):
                rel_desc = relationships[i].type.value.lower().replace('_', ' ')
                path_desc.append(f"--({rel_desc})-->")
        
        return " ".join(path_desc)
    
    async def get_entity_neighborhood(self, node_id: str, 
                                   max_depth: int = 2,
                                   max_neighbors: int = 50) -> EntityNeighborhood:
        """
        Get comprehensive neighborhood analysis for an entity.
        
        Args:
            node_id: ID of the central node
            max_depth: Maximum depth to explore
            max_neighbors: Maximum number of neighbors to return
            
        Returns:
            EntityNeighborhood: Comprehensive neighborhood analysis
        """
        try:
            # Get central node
            central_node = await self.graph_store.get_node(node_id)
            if not central_node:
                raise ValueError(f"Node {node_id} not found")
            
            # Get all neighbors within max_depth
            cypher = f"""
            MATCH (center) WHERE elementId(center) = $node_id
            MATCH path = (center)-[*1..{max_depth}]-(neighbor)
            WHERE ALL(rel in relationships(path) WHERE rel.confidence >= $confidence)
            RETURN neighbor, relationships(path) as rels, length(path) as depth
            ORDER BY depth, neighbor.name
            LIMIT $max_neighbors
            """
            
            cypher_query = GraphQuery(
                cypher=cypher,
                parameters={
                    "node_id": node_id,
                    "confidence": self.confidence_threshold,
                    "max_neighbors": max_neighbors
                }
            )
            
            result = await self.graph_store.query(cypher_query)
            
            # Process results
            neighbors = []
            all_relationships = []
            neighbor_types = {}
            relationship_types = {}
            depth_analysis = {}
            
            for node in result.nodes:
                if node.id != node_id:
                    neighbors.append(node)
                    
                    # Count node types
                    node_type = node.type.value
                    neighbor_types[node_type] = neighbor_types.get(node_type, 0) + 1
            
            for rel in result.relationships:
                all_relationships.append(rel)
                
                # Count relationship types
                rel_type = rel.type.value
                relationship_types[rel_type] = relationship_types.get(rel_type, 0) + 1
            
            # Get direct neighbors only
            direct_neighbors = []
            for node in neighbors:
                # Check if it's a direct neighbor (this is simplified)
                # In a real implementation, you'd track the depth from the query results
                direct_neighbors.append(node)
            
            # Build depth analysis (simplified)
            depth_analysis = {1: len(direct_neighbors)}
            
            return EntityNeighborhood(
                central_node=central_node,
                direct_neighbors=direct_neighbors,
                relationships=all_relationships,
                neighbor_types=neighbor_types,
                relationship_types=relationship_types,
                depth_analysis=depth_analysis
            )
            
        except Exception as e:
            logger.error(f"Failed to get entity neighborhood for {node_id}: {e}")
            # Return empty neighborhood
            empty_node = GraphNode(name="unknown")
            return EntityNeighborhood(
                central_node=empty_node,
                direct_neighbors=[],
                relationships=[],
                neighbor_types={},
                relationship_types={},
                depth_analysis={}
            )
    
    async def find_connection_paths(self, entity1_name: str, entity2_name: str,
                                  max_depth: int = 4) -> List[GraphPath]:
        """
        Find connection paths between two named entities.
        
        Args:
            entity1_name: Name of the first entity
            entity2_name: Name of the second entity
            max_depth: Maximum path length to search
            
        Returns:
            List[GraphPath]: Paths connecting the entities
        """
        try:
            # Find nodes by name
            nodes1 = await self.graph_store.find_nodes_by_property("name", entity1_name)
            nodes2 = await self.graph_store.find_nodes_by_property("name", entity2_name)
            
            if not nodes1 or not nodes2:
                logger.warning(f"Could not find nodes for {entity1_name} or {entity2_name}")
                return []
            
            # Use the first matching node for each entity
            node1_id = nodes1[0].id
            node2_id = nodes2[0].id
            
            # Create traversal query
            query = TraversalQuery(
                start_node_id=node1_id,
                end_node_id=node2_id,
                pattern=PathPattern.ALL_PATHS,
                max_depth=max_depth,
                max_results=10
            )
            
            result = await self.relationship_search(query)
            return result.paths
            
        except Exception as e:
            logger.error(f"Failed to find connection paths: {e}")
            return []

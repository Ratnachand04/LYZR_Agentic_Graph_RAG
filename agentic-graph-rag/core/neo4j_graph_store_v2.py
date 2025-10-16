#!/usr/bin/env python3
"""
Neo4j Graph Store Integration for Agentic Graph RAG Pipeline
Handles persistent storage and retrieval of knowledge graphs in Neo4j database.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import json

try:
    from neo4j import GraphDatabase
    from neo4j.exceptions import ServiceUnavailable, AuthError
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    GraphDatabase = None

logger = logging.getLogger(__name__)


@dataclass
class Neo4jConfig:
    """Configuration for Neo4j connection."""
    uri: str = "bolt://localhost:7687"
    username: str = "neo4j"
    password: str = "password"
    database: str = "agentic_graph_rag"
    max_retry_time: int = 30
    pool_size: int = 20


class Neo4jGraphStore:
    """
    Neo4j integration for storing and querying knowledge graphs.
    
    Features:
    - Automatic schema creation
    - Batch operations for performance
    - Entity and relationship management
    - Embedding storage and similarity search
    - Graph traversal and analysis
    """
    
    def __init__(self, config: Neo4jConfig = None):
        """Initialize Neo4j connection."""
        self.config = config or Neo4jConfig()
        self.driver = None
        self.connected = False
        
        if not NEO4J_AVAILABLE:
            logger.warning("Neo4j Python driver not available. Install with: pip install neo4j")
            return
            
        self._connect()
    
    def _connect(self):
        """Establish connection to Neo4j database."""
        try:
            # Use driver-supported parameters for connection pooling and lifetime
            self.driver = GraphDatabase.driver(
                self.config.uri,
                auth=(self.config.username, self.config.password),
                max_connection_lifetime=self.config.max_retry_time,
                max_connection_pool_size=self.config.pool_size
            )
            
            # Test connection
            with self.driver.session(database=self.config.database) as session:
                result = session.run("RETURN 1")
                result.single()
            
            self.connected = True
            logger.info(f"Connected to Neo4j at {self.config.uri}")
            
        except ServiceUnavailable:
            logger.error(f"Neo4j not available at {self.config.uri}")
            logger.info("To start Neo4j:")
            logger.info("1. Install Neo4j Desktop: https://neo4j.com/download/")
            logger.info("2. Or use Docker: docker run -p 7474:7474 -p 7687:7687 neo4j:latest")
            self.connected = False
            
        except AuthError:
            logger.error("Neo4j authentication failed")
            logger.info("Default credentials: neo4j/password")
            self.connected = False
            
        except Exception as e:
            logger.error(f"Neo4j connection failed: {e}")
            self.connected = False
    
    async def initialize_schema(self) -> Dict[str, Any]:
        """
        Create indexes and constraints for optimal performance.
        """
        if not self.connected:
            return {"status": "error", "message": "Not connected to Neo4j"}
        
        schema_commands = [
            # Entity constraints and indexes
            "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
            "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)",
            "CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type)",
            
            # Relationship indexes
            "CREATE INDEX rel_type IF NOT EXISTS FOR ()-[r:RELATES_TO]-() ON (r.type)",
            "CREATE INDEX rel_confidence IF NOT EXISTS FOR ()-[r:RELATES_TO]-() ON (r.confidence)",
            
            # Document tracking
            "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
            "CREATE INDEX document_path IF NOT EXISTS FOR (d:Document) ON (d.path)",
            
            # Session tracking
            "CREATE INDEX session_id IF NOT EXISTS FOR (s:Session) ON (s.session_id)",
            
            # Embedding indexes (if using vector search)
            "CREATE VECTOR INDEX entity_embeddings IF NOT EXISTS FOR (e:Entity) ON (e.embedding) OPTIONS {indexConfig: {`vector.dimensions`: 1536, `vector.similarity_function`: 'cosine'}}",
        ]
        
        results = []
        with self.driver.session(database=self.config.database) as session:
            for command in schema_commands:
                try:
                    result = session.run(command)
                    results.append({"command": command, "status": "success"})
                    logger.debug(f"Schema command executed: {command}")
                except Exception as e:
                    results.append({"command": command, "status": "error", "error": str(e)})
                    logger.warning(f"Schema command failed: {command} - {e}")
        
        logger.info("Neo4j schema initialization completed")
        return {
            "status": "success",
            "schema_commands_executed": len([r for r in results if r["status"] == "success"]),
            "schema_commands_failed": len([r for r in results if r["status"] == "error"]),
            "details": results
        }
    
    async def store_entities(self, entities: List[Dict[str, Any]], session_id: str) -> Dict[str, Any]:
        """
        Store entities in Neo4j with batch processing for performance.
        """
        if not self.connected:
            return {"status": "error", "message": "Not connected to Neo4j"}
        
        if not entities:
            return {"status": "success", "entities_stored": 0}
        
        # Batch processing for large entity sets
        batch_size = 1000
        total_stored = 0
        
        with self.driver.session(database=self.config.database) as neo4j_session:
            for i in range(0, len(entities), batch_size):
                batch = entities[i:i + batch_size]
                
                # Prepare entity data
                entity_data = []
                for entity in batch:
                    entity_record = {
                        'id': entity['id'],
                        'name': entity['name'],
                        'type': entity['type'],
                        'session_id': session_id,
                        'confidence': entity.get('confidence', 0.0),
                        'source_file': entity.get('source_file', ''),
                        'created_at': datetime.utcnow().isoformat(),
                        'properties': json.dumps(entity.get('properties', {}))
                    }
                    
                    # Add embedding if available
                    if 'embedding' in entity and entity['embedding']:
                        entity_record['embedding'] = entity['embedding']
                    
                    entity_data.append(entity_record)
                
                # Batch insert
                cypher_query = """
                UNWIND $entities AS entity
                MERGE (e:Entity {id: entity.id})
                SET e.name = entity.name,
                    e.type = entity.type,
                    e.session_id = entity.session_id,
                    e.confidence = entity.confidence,
                    e.source_file = entity.source_file,
                    e.created_at = entity.created_at,
                    e.properties = entity.properties
                """
                
                # Add embedding clause if embeddings are present
                if any('embedding' in record for record in entity_data):
                    cypher_query += """
                    SET e.embedding = CASE 
                        WHEN entity.embedding IS NOT NULL 
                        THEN entity.embedding 
                        ELSE e.embedding 
                    END
                    """
                
                try:
                    result = neo4j_session.run(cypher_query, entities=entity_data)
                    summary = result.consume()
                    total_stored += summary.counters.nodes_created + summary.counters.properties_set
                    
                except Exception as e:
                    logger.error(f"Failed to store entity batch: {e}")
                    return {"status": "error", "message": str(e), "entities_stored": total_stored}
        
        logger.info(f"Stored {len(entities)} entities in Neo4j")
        return {
            "status": "success", 
            "entities_stored": len(entities),
            "total_operations": total_stored
        }
    
    async def store_relationships(self, relationships: List[Dict[str, Any]], session_id: str) -> Dict[str, Any]:
        """
        Store relationships in Neo4j with batch processing.
        """
        if not self.connected:
            return {"status": "error", "message": "Not connected to Neo4j"}
        
        if not relationships:
            return {"status": "success", "relationships_stored": 0}
        
        batch_size = 1000
        total_stored = 0
        
        with self.driver.session(database=self.config.database) as neo4j_session:
            for i in range(0, len(relationships), batch_size):
                batch = relationships[i:i + batch_size]
                
                # Prepare relationship data
                rel_data = []
                for rel in batch:
                    rel_record = {
                        'id': rel['id'],
                        'source_id': rel['source'],
                        'target_id': rel['target'],
                        'type': rel['type'],
                        'session_id': session_id,
                        'confidence': rel.get('confidence', 0.0),
                        'source_file': rel.get('source_file', ''),
                        'created_at': datetime.utcnow().isoformat(),
                        'properties': json.dumps(rel.get('properties', {}))
                    }
                    rel_data.append(rel_record)
                
                # Batch insert relationships
                cypher_query = """
                UNWIND $relationships AS rel
                MATCH (source:Entity {id: rel.source_id})
                MATCH (target:Entity {id: rel.target_id})
                MERGE (source)-[r:RELATES_TO {id: rel.id}]->(target)
                SET r.type = rel.type,
                    r.session_id = rel.session_id,
                    r.confidence = rel.confidence,
                    r.source_file = rel.source_file,
                    r.created_at = rel.created_at,
                    r.properties = rel.properties
                """
                
                try:
                    result = neo4j_session.run(cypher_query, relationships=rel_data)
                    summary = result.consume()
                    total_stored += summary.counters.relationships_created + summary.counters.properties_set
                    
                except Exception as e:
                    logger.error(f"Failed to store relationship batch: {e}")
                    return {"status": "error", "message": str(e), "relationships_stored": total_stored}
        
        logger.info(f"Stored {len(relationships)} relationships in Neo4j")
        return {
            "status": "success", 
            "relationships_stored": len(relationships),
            "total_operations": total_stored
        }
    
    async def get_graph_statistics(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the stored graph.
        """
        if not self.connected:
            return {"status": "error", "message": "Not connected to Neo4j"}
        
        session_filter = f"WHERE e.session_id = '{session_id}'" if session_id else ""
        
        queries = {
            'total_entities': f"MATCH (e:Entity) {session_filter} RETURN count(e) as count",
            'total_relationships': f"MATCH ()-[r:RELATES_TO]->() {session_filter.replace('e.', 'r.')} RETURN count(r) as count",
            'entity_types': f"MATCH (e:Entity) {session_filter} RETURN e.type as type, count(e) as count ORDER BY count DESC",
            'relationship_types': f"MATCH ()-[r:RELATES_TO]->() {session_filter.replace('e.', 'r.')} RETURN r.type as type, count(r) as count ORDER BY count DESC",
            'sessions': "MATCH (e:Entity) RETURN DISTINCT e.session_id as session_id, count(e) as entities ORDER BY entities DESC",
        }
        
        stats = {}
        
        with self.driver.session(database=self.config.database) as neo4j_session:
            for query_name, cypher_query in queries.items():
                try:
                    result = neo4j_session.run(cypher_query)
                    if query_name in ['total_entities', 'total_relationships']:
                        stats[query_name] = result.single()['count']
                    else:
                        stats[query_name] = [record.data() for record in result]
                        
                except Exception as e:
                    logger.error(f"Statistics query failed ({query_name}): {e}")
                    stats[query_name] = None
        
        return {
            "status": "success",
            "session_id": session_id,
            "statistics": stats,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def similarity_search(self, 
                              query_embedding: List[float], 
                              limit: int = 10,
                              similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search on entity embeddings.
        """
        if not self.connected:
            return []
        
        # Neo4j vector similarity search
        cypher_query = """
        MATCH (e:Entity)
        WHERE e.embedding IS NOT NULL
        CALL db.index.vector.queryNodes('entity_embeddings', $limit, $query_embedding)
        YIELD node, score
        WHERE score >= $similarity_threshold
        RETURN node.id as id, node.name as name, node.type as type, 
               node.confidence as confidence, score
        ORDER BY score DESC
        """
        
        try:
            with self.driver.session(database=self.config.database) as session:
                result = session.run(
                    cypher_query, 
                    query_embedding=query_embedding,
                    limit=limit,
                    similarity_threshold=similarity_threshold
                )
                
                return [record.data() for record in result]
                
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []
    
    async def graph_traversal(self, 
                            start_entity_id: str,
                            max_depth: int = 3,
                            relationship_filter: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform graph traversal starting from a specific entity.
        """
        if not self.connected:
            return {"nodes": [], "edges": []}
        
        # Build relationship filter
        rel_filter = f"WHERE r.type = '{relationship_filter}'" if relationship_filter else ""
        
        cypher_query = f"""
        MATCH path = (start:Entity {{id: $start_id}})-[r:RELATES_TO*1..{max_depth}]-(connected:Entity)
        {rel_filter}
        RETURN nodes(path) as nodes, relationships(path) as relationships
        """
        
        try:
            with self.driver.session(database=self.config.database) as session:
                result = session.run(cypher_query, start_id=start_entity_id)
                
                nodes = set()
                edges = []
                
                for record in result:
                    path_nodes = record['nodes']
                    path_rels = record['relationships']
                    
                    # Collect nodes
                    for node in path_nodes:
                        nodes.add((node['id'], node['name'], node['type']))
                    
                    # Collect relationships
                    for rel in path_rels:
                        edges.append({
                            'source': rel.start_node['id'],
                            'target': rel.end_node['id'],
                            'type': rel['type'],
                            'confidence': rel.get('confidence', 0.0)
                        })
                
                return {
                    "nodes": [{"id": n[0], "name": n[1], "type": n[2]} for n in nodes],
                    "edges": edges,
                    "start_entity": start_entity_id,
                    "max_depth": max_depth
                }
                
        except Exception as e:
            logger.error(f"Graph traversal failed: {e}")
            return {"nodes": [], "edges": [], "error": str(e)}
    
    def close(self):
        """Close Neo4j connection."""
        if self.driver:
            self.driver.close()
            self.connected = False
            logger.info("Neo4j connection closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Development helper functions
async def setup_development_neo4j():
    """
    Helper function to set up Neo4j for development.
    Provides instructions and testing capabilities.
    """
    logger.info("Setting up Neo4j for development...")
    
    # Check if Neo4j is available
    config = Neo4jConfig()
    store = Neo4jGraphStore(config)
    
    if store.connected:
        logger.info("✅ Neo4j is connected and ready")
        
        # Initialize schema
        schema_result = await store.initialize_schema()
        logger.info(f"Schema initialization: {schema_result['status']}")
        
        # Get current statistics
        stats = await store.get_graph_statistics()
        if stats['status'] == 'success':
            logger.info(f"Current graph: {stats['statistics']['total_entities']} entities, {stats['statistics']['total_relationships']} relationships")
        
        store.close()
        return True
        
    else:
        logger.error("❌ Neo4j is not available")
        logger.info("\nTo set up Neo4j for development:")
        logger.info("1. Option 1 - Neo4j Desktop:")
        logger.info("   - Download: https://neo4j.com/download/")
        logger.info("   - Create new database with password 'password'")
        logger.info("   - Start the database")
        logger.info("\n2. Option 2 - Docker:")
        logger.info("   docker run -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:latest")
        logger.info("\n3. Option 3 - Cloud (Neo4j Aura):")
        logger.info("   - Sign up at https://neo4j.com/cloud/aura/")
        logger.info("   - Update connection details in config")
        
        return False


if __name__ == "__main__":
    # Test Neo4j connection and setup
    import asyncio
    
    async def test_connection():
        success = await setup_development_neo4j()
        if success:
            print("✅ Neo4j setup completed successfully")
        else:
            print("❌ Neo4j setup failed - see instructions above")
    
    asyncio.run(test_connection())
#!/usr/bin/env python3
"""
Phase 2 - Production Agentic Graph RAG System
Main pipeline for document-to-graph processing with LLM ontology generation and Neo4j integration.
"""

import asyncio
import json
import logging
import sys
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import click
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Core components
from core.simple_document_processor import DocumentProcessor
from core.simple_graph_visualizer import SimpleGraphVisualizer

# Phase 2 components (to be implemented)
# from core.llm_ontology_generator import LLMOntologyGenerator
# from core.neo4j_graph_store import Neo4jGraphStore
# from core.embedding_manager import EmbeddingManager
# from core.entity_resolver import EntityResolver

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AgenticGraphRAGPipeline:
    """
    Production-grade Document-to-Graph RAG Pipeline with LLM ontology generation.
    
    Pipeline Flow:
    1. Document Ingestion → Multi-format text extraction
    2. LLM Ontology Generation → Structured entities and relationships
    3. Embedding Integration → Vector representations for all graph elements
    4. Entity Resolution → Deduplication and merging
    5. Neo4j Storage → Persistent graph database
    6. 3D Visualization → Interactive graph exploration
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the production pipeline."""
        self.config = config or {}
        
        # Phase 1 components (proven and working)
        self.document_processor = DocumentProcessor()
        self.graph_visualizer = SimpleGraphVisualizer()
        
        # Phase 2 components (to be integrated)
        self.llm_ontology_generator = None
        self.neo4j_store = None
        self.embedding_manager = None
        self.entity_resolver = None
        
        # Processing state
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.processed_documents = {}
        self.ontology_schema = {}
        self.knowledge_graph = {
            'entities': [],
            'relationships': [],
            'embeddings': {},
            'metadata': {}
        }
        
        logger.info(f"Initialized Agentic Graph RAG Pipeline (Session: {self.session_id})")
    
    async def run_full_pipeline(self, 
                               input_paths: List[str], 
                               output_path: str = None,
                               enable_llm: bool = True,
                               enable_neo4j: bool = True,
                               enable_embeddings: bool = True) -> Dict[str, Any]:
        """
        Execute the complete document-to-graph pipeline.
        
        Args:
            input_paths: List of document file paths or directories
            output_path: Output path for visualization
            enable_llm: Use LLM for ontology generation
            enable_neo4j: Store in Neo4j database
            enable_embeddings: Generate embeddings for entities
            
        Returns:
            Pipeline results with statistics and outputs
        """
        logger.info("Starting Agentic Graph RAG Pipeline")
        start_time = datetime.now()
        
        results = {
            'session_id': self.session_id,
            'start_time': start_time.isoformat(),
            'input_files': [],
            'pipeline_stages': {},
            'knowledge_graph': {},
            'outputs': {},
            'statistics': {}
        }
        
        try:
            # Stage 1: Document Ingestion and Text Extraction
            stage_1_result = await self._stage_1_document_ingestion(input_paths)
            results['pipeline_stages']['document_ingestion'] = stage_1_result
            results['input_files'] = stage_1_result['processed_files']
            
            # Stage 2: LLM Ontology Generation
            if enable_llm:
                stage_2_result = await self._stage_2_llm_ontology_generation(stage_1_result['extracted_texts'])
                results['pipeline_stages']['ontology_generation'] = stage_2_result
                self.ontology_schema = stage_2_result['ontology_schema']
            else:
                # Fallback to simple entity extraction
                stage_2_result = await self._stage_2_simple_extraction(stage_1_result['extracted_texts'])
                results['pipeline_stages']['simple_extraction'] = stage_2_result
            
            # Stage 3: Embedding Integration
            if enable_embeddings:
                stage_3_result = await self._stage_3_embedding_integration()
                results['pipeline_stages']['embedding_integration'] = stage_3_result
            
            # Stage 4: Entity Resolution and Deduplication
            stage_4_result = await self._stage_4_entity_resolution()
            results['pipeline_stages']['entity_resolution'] = stage_4_result
            
            # Stage 5: Neo4j Graph Storage
            if enable_neo4j:
                stage_5_result = await self._stage_5_neo4j_storage()
                results['pipeline_stages']['neo4j_storage'] = stage_5_result
            
            # Stage 6: Graph Visualization
            stage_6_result = await self._stage_6_visualization(output_path)
            results['pipeline_stages']['visualization'] = stage_6_result
            results['outputs']['visualization_path'] = stage_6_result['output_path']
            
            # Final statistics
            end_time = datetime.now()
            results['end_time'] = end_time.isoformat()
            results['total_duration'] = str(end_time - start_time)
            results['knowledge_graph'] = self.knowledge_graph
            results['statistics'] = self._generate_statistics()
            
            logger.info(f"Pipeline completed successfully in {end_time - start_time}")
            return results
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            results['error'] = str(e)
            results['status'] = 'failed'
            return results
    
    async def _stage_1_document_ingestion(self, input_paths: List[str]) -> Dict[str, Any]:
        """Stage 1: Multi-format document processing and text extraction."""
        logger.info("Stage 1: Document Ingestion and Text Extraction")
        
        # Collect all files
        file_paths = []
        for path in input_paths:
            path_obj = Path(path)
            if path_obj.is_file():
                file_paths.append(path_obj)
            elif path_obj.is_dir():
                # Recursively find supported files
                for pattern in ['*.pdf', '*.docx', '*.txt', '*.md', '*.jpg', '*.png']:
                    file_paths.extend(path_obj.rglob(pattern))
        
        logger.info(f"Found {len(file_paths)} files to process")
        
        extracted_texts = {}
        processed_files = []
        failed_files = []
        
        for file_path in file_paths:
            try:
                logger.info(f"Processing: {file_path.name}")
                result = await self.document_processor.extract_text(file_path)
                
                if result['success']:
                    extracted_texts[str(file_path)] = {
                        'text': result['text'],
                        'metadata': result['metadata'],
                        'method': result['method']
                    }
                    processed_files.append({
                        'path': str(file_path),
                        'size': len(result['text']),
                        'method': result['method']
                    })
                    logger.info(f"✅ Extracted {len(result['text'])} characters")
                else:
                    failed_files.append({
                        'path': str(file_path),
                        'error': result.get('error', 'Unknown error')
                    })
                    logger.warning(f"❌ Failed: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                failed_files.append({'path': str(file_path), 'error': str(e)})
                logger.error(f"❌ Exception processing {file_path}: {e}")
        
        self.processed_documents = extracted_texts
        
        return {
            'processed_files': processed_files,
            'failed_files': failed_files,
            'extracted_texts': extracted_texts,
            'total_files': len(file_paths),
            'success_rate': len(processed_files) / len(file_paths) if file_paths else 0,
            'total_characters': sum(len(data['text']) for data in extracted_texts.values())
        }
    
    async def _stage_2_llm_ontology_generation(self, extracted_texts: Dict[str, Any]) -> Dict[str, Any]:
        """Stage 2: LLM-powered ontology generation with structured entities and relationships."""
        logger.info("Stage 2: LLM Ontology Generation")
        
        # TODO: Implement LLM ontology generation
        # This will use OpenRouter API to generate structured ontologies
        logger.warning("LLM Ontology Generation not yet implemented - using fallback")
        return await self._stage_2_simple_extraction(extracted_texts)
    
    async def _stage_2_simple_extraction(self, extracted_texts: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback: Simple pattern-based entity and relationship extraction."""
        logger.info("Stage 2: Simple Entity/Relationship Extraction (Fallback)")
        
        # Use existing entity extraction logic from Phase 1
        entities = []
        relationships = []
        
        for file_path, data in extracted_texts.items():
            text = data['text']
            
            # Enhanced entity extraction patterns
            entity_patterns = {
                'person': r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',
                'organization': r'\b(?:Company|Corp|Inc|Ltd|LLC|Organization|University|Institute)\b[^.]*',
                'concept': r'\b(?:theory|method|approach|technique|system|model|framework)\b[^.]*',
                'location': r'\b(?:in|at|from) ([A-Z][a-z]+(?: [A-Z][a-z]+)*)\b'
            }
            
            file_entities = []
            for entity_type, pattern in entity_patterns.items():
                import re
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches[:10]:  # Limit to prevent explosion
                    entity = {
                        'id': f"{entity_type}_{len(entities)}",
                        'name': match.strip(),
                        'type': entity_type,
                        'source_file': file_path,
                        'confidence': 0.7
                    }
                    entities.append(entity)
                    file_entities.append(entity)
            
            # Simple relationship extraction
            for i, entity1 in enumerate(file_entities):
                for entity2 in file_entities[i+1:]:
                    if entity1['type'] != entity2['type']:  # Different types likely to have relationships
                        relationship = {
                            'id': f"rel_{len(relationships)}",
                            'source': entity1['id'],
                            'target': entity2['id'],
                            'type': f"{entity1['type']}_to_{entity2['type']}",
                            'source_file': file_path,
                            'confidence': 0.5
                        }
                        relationships.append(relationship)
        
        self.knowledge_graph['entities'] = entities
        self.knowledge_graph['relationships'] = relationships
        
        return {
            'entities_extracted': len(entities),
            'relationships_extracted': len(relationships),
            'ontology_schema': {
                'entity_types': list(set(e['type'] for e in entities)),
                'relationship_types': list(set(r['type'] for r in relationships))
            }
        }
    
    async def _stage_3_embedding_integration(self) -> Dict[str, Any]:
        """Stage 3: Generate embeddings for all graph elements."""
        logger.info("Stage 3: Embedding Integration")
        
        # TODO: Implement OpenAI embedding integration
        logger.warning("Embedding Integration not yet implemented")
        return {
            'embeddings_generated': 0,
            'embedding_model': 'text-embedding-ada-002',
            'status': 'not_implemented'
        }
    
    async def _stage_4_entity_resolution(self) -> Dict[str, Any]:
        """Stage 4: Entity deduplication and merging using embeddings."""
        logger.info("Stage 4: Entity Resolution and Deduplication")
        
        # TODO: Implement embedding-based entity resolution
        logger.warning("Entity Resolution not yet implemented")
        return {
            'entities_before_resolution': len(self.knowledge_graph['entities']),
            'entities_after_resolution': len(self.knowledge_graph['entities']),
            'duplicates_merged': 0,
            'status': 'not_implemented'
        }
    
    async def _stage_5_neo4j_storage(self) -> Dict[str, Any]:
        """Stage 5: Store knowledge graph in Neo4j database."""
        logger.info("Stage 5: Neo4j Graph Storage")
        
        # TODO: Implement Neo4j integration
        logger.warning("Neo4j Integration not yet implemented")
        return {
            'nodes_stored': 0,
            'relationships_stored': 0,
            'database_url': 'bolt://localhost:7687',
            'status': 'not_implemented'
        }
    
    async def _stage_6_visualization(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        """Stage 6: Generate interactive 3D graph visualization."""
        logger.info("Stage 6: Graph Visualization")
        
        if not output_path:
            output_path = f"agentic_graph_{self.session_id}.html"
        
        # Use existing visualization from Phase 1
        entities = self.knowledge_graph['entities']
        relationships = self.knowledge_graph['relationships']
        
        if not entities:
            logger.warning("No entities to visualize")
            return {'output_path': output_path, 'status': 'no_data'}
        
        try:
            # Use the synchronous method that works with entity/relationship format
            html_content = self.graph_visualizer.create_interactive_graph(
                entities=entities,
                relationships=relationships,
                title=f"Agentic Knowledge Graph - Session {self.session_id}"
            )
            
            # Save to file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            viz_result = output_path
            
            logger.info(f"Visualization saved to: {output_path}")
            return {
                'output_path': output_path,
                'nodes_visualized': len(entities),
                'edges_visualized': len(relationships),
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Visualization failed: {e}")
            return {'output_path': output_path, 'status': 'failed', 'error': str(e)}
    
    def _generate_statistics(self) -> Dict[str, Any]:
        """Generate comprehensive pipeline statistics."""
        return {
            'documents_processed': len(self.processed_documents),
            'entities_extracted': len(self.knowledge_graph['entities']),
            'relationships_extracted': len(self.knowledge_graph['relationships']),
            'entity_types': len(set(e['type'] for e in self.knowledge_graph['entities'])),
            'relationship_types': len(set(r['type'] for r in self.knowledge_graph['relationships'])),
            'total_text_characters': sum(
                len(data['text']) for data in self.processed_documents.values()
            ),
            'session_id': self.session_id
        }


@click.command()
@click.argument('input_paths', nargs=-1, required=True)
@click.option('--output', '-o', default=None, help='Output path for visualization')
@click.option('--no-llm', is_flag=True, help='Disable LLM ontology generation')
@click.option('--no-neo4j', is_flag=True, help='Disable Neo4j storage')
@click.option('--no-embeddings', is_flag=True, help='Disable embeddings')
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file')
@click.option('--debug', is_flag=True, help='Enable debug logging')
def main(input_paths, output, no_llm, no_neo4j, no_embeddings, config, debug):
    """
    Agentic Graph RAG Pipeline - Process documents into knowledge graphs.
    
    INPUT_PATHS: One or more file paths or directories to process
    """
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load configuration if provided
    config_data = {}
    if config:
        with open(config, 'r') as f:
            config_data = json.load(f)
    
    async def run_pipeline():
        pipeline = AgenticGraphRAGPipeline(config_data)
        
        results = await pipeline.run_full_pipeline(
            input_paths=list(input_paths),
            output_path=output,
            enable_llm=not no_llm,
            enable_neo4j=not no_neo4j,
            enable_embeddings=not no_embeddings
        )
        
        # Display results
        print("\n" + "="*60)
        print("AGENTIC GRAPH RAG PIPELINE RESULTS")
        print("="*60)
        print(f"Session ID: {results['session_id']}")
        print(f"Duration: {results.get('total_duration', 'N/A')}")
        print(f"Documents Processed: {results['statistics']['documents_processed']}")
        print(f"Entities Extracted: {results['statistics']['entities_extracted']}")
        print(f"Relationships Found: {results['statistics']['relationships_extracted']}")
        
        if 'visualization_path' in results.get('outputs', {}):
            print(f"Visualization: {results['outputs']['visualization_path']}")
            
            # Open visualization in browser
            try:
                import webbrowser
                webbrowser.open(f"file://{Path(results['outputs']['visualization_path']).absolute()}")
            except Exception as e:
                logger.warning(f"Could not open browser: {e}")
        
        print("="*60)
        
        if results.get('status') == 'failed':
            sys.exit(1)
    
    asyncio.run(run_pipeline())


if __name__ == "__main__":
    main()
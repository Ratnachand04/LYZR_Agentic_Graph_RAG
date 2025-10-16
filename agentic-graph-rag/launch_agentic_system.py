#!/usr/bin/env python3
"""
Complete Agentic Graph RAG System - Main Launch Script
Launch the full agentic system with all components integrated.
"""

import os
import sys
import asyncio
import logging
import argparse
import threading
import time
from pathlib import Path
from typing import Optional, Dict, Any
import signal
import json

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import configuration
from config import AgenticRAGConfig, get_development_config, get_production_config

# Import core components
try:
    from core.faiss_index_manager import FAISSIndexManager
    from core.cypher_traversal_engine import CypherTraversalEngine
    from core.query_analyzer import QueryAnalyzer
    from core.reasoning_engine_unified import ReasoningEngine
    from core.interactive_graph_editor import InteractiveGraphEditor
    from core.agentic_rag_api import AgenticRAGAPI
    from core.agentic_rag_gui import AgenticRAGGUI
    from core.neo4j_graph_store_v2 import Neo4jGraphStore
    from core.embedding_manager_v2 import EmbeddingManager
except ImportError as e:
    print(f"Warning: Some components not available: {e}")
    print("This is expected if you haven't installed all dependencies yet.")

logger = logging.getLogger(__name__)


class AgenticSystemManager:
    """
    Main system manager for the complete agentic RAG system.
    
    Coordinates all components and provides unified control interface.
    """
    
    def __init__(self, config: AgenticRAGConfig):
        """Initialize system manager."""
        self.config = config
        self.components = {}
        self.running = False
        self.startup_tasks = []
        
        # Setup logging
        self.config.setup_logging()
        
        logger.info("Agentic System Manager initialized")
    
    async def initialize_components(self):
        """Initialize all system components."""
        logger.info("Initializing agentic system components...")
        
        try:
            # Create Neo4j configuration
            from core.neo4j_graph_store_v2 import Neo4jConfig
            neo4j_config = Neo4jConfig(
                uri=self.config.neo4j.uri,
                username=self.config.neo4j.username,
                password=self.config.neo4j.password,
                database=self.config.neo4j.database
            )
            
            # Initialize Neo4j Graph Store
            self.components['graph_store'] = Neo4jGraphStore(config=neo4j_config)
            logger.info("✓ Neo4j Graph Store initialized")
            
            # Create Embedding configuration
            from core.embedding_manager_v2 import EmbeddingConfig
            embedding_config = EmbeddingConfig(
                api_key=self.config.api.openai_api_key or self.config.api.openrouter_api_key,
                model=self.config.api.embedding_model,
                base_url=self.config.api.openrouter_base_url if self.config.api.openrouter_api_key else "https://api.openai.com/v1"
            )
            
            # Initialize Embedding Manager
            self.components['embedding_manager'] = EmbeddingManager(config=embedding_config)
            logger.info("✓ Embedding Manager initialized")
            
            # Create FAISS configuration
            from core.faiss_index_manager import IndexConfig
            faiss_config = IndexConfig(
                dimension=self.config.faiss.dimension,
                index_type=self.config.faiss.index_type
            )
            
            # Initialize FAISS Index Manager
            self.components['faiss_manager'] = FAISSIndexManager(
                config=faiss_config,
                index_dir=self.config.faiss.index_path
            )
            logger.info("✓ FAISS Index Manager initialized")
            
            # Initialize Cypher Traversal Engine
            self.components['cypher_engine'] = CypherTraversalEngine(
                neo4j_uri=self.config.neo4j.uri,
                neo4j_user=self.config.neo4j.username,
                neo4j_password=self.config.neo4j.password
            )
            logger.info("✓ Cypher Traversal Engine initialized")
            
            # Initialize Query Analyzer
            self.components['query_analyzer'] = QueryAnalyzer()
            logger.info("✓ Query Analyzer initialized")
            
            # Initialize Reasoning Engine
            self.components['reasoning_engine'] = ReasoningEngine(
                vector_index=self.components['faiss_manager'],
                graph_engine=self.components['cypher_engine']
            )
            logger.info("✓ Reasoning Engine initialized")
            
            # Initialize Interactive Graph Editor
            self.components['graph_editor'] = InteractiveGraphEditor()
            logger.info("✓ Interactive Graph Editor initialized")
            
            # Initialize API Server
            self.components['api_server'] = AgenticRAGAPI(
                vector_index=self.components['faiss_manager'],
                graph_engine=self.components['cypher_engine'],
                reasoning_engine=self.components['reasoning_engine']
            )
            logger.info("✓ Agentic RAG API initialized")
            
            logger.info(f"🎉 All {len(self.components)} components initialized successfully!")
            
        except Exception as e:
            logger.error(f"❌ Component initialization failed: {e}")
            raise
    
    def start_web_server(self):
        """Start the web API server."""
        try:
            import uvicorn
            from core.agentic_rag_api import AgenticRAGAPI
            import core.agentic_rag_api as api_module
            
            logger.info("🚀 Starting Agentic RAG API server...")
            
            # Create API instance and set module-level app
            api_instance = AgenticRAGAPI(
                vector_index=self.components['faiss_manager'],
                graph_engine=self.components['cypher_engine'],
                reasoning_engine=self.components['reasoning_engine']
            )
            api_module.app = api_instance.app
            
            # Run server in separate thread
            def run_server():
                uvicorn.run(
                    api_instance.app,
                    host=self.config.web_server.host,
                    port=self.config.web_server.port,
                    log_level=self.config.system.log_level.lower()
                )
            
            server_thread = threading.Thread(target=run_server, daemon=True)
            server_thread.start()
            
            logger.info(f"✓ API server started on http://{self.config.web_server.host}:{self.config.web_server.port}")
            
        except Exception as e:
            logger.error(f"❌ Failed to start web server: {e}")
            raise
    
    def start_gui(self):
        """Start the GUI application."""
        try:
            logger.info("🖥️  Starting Agentic RAG GUI...")
            
            # Initialize and run GUI
            gui_app = AgenticRAGGUI()
            gui_app.run()
            
        except Exception as e:
            logger.error(f"❌ Failed to start GUI: {e}")
            raise
    
    async def start_system(self, mode: str = "full"):
        """Start the complete agentic system."""
        logger.info(f"🚀 Starting Agentic RAG System in '{mode}' mode...")
        
        try:
            # Initialize components
            await self.initialize_components()
            
            # Test connections
            await self.test_connections()
            
            self.running = True
            
            if mode == "api":
                # Start only API server
                self.start_web_server()
                logger.info("✅ Agentic RAG System (API mode) started successfully!")
                
                # Keep running
                try:
                    while self.running:
                        await asyncio.sleep(1)
                except KeyboardInterrupt:
                    logger.info("Shutting down...")
            
            elif mode == "gui":
                # Start only GUI
                self.start_gui()
                logger.info("✅ Agentic RAG System (GUI mode) started successfully!")
            
            elif mode == "full":
                # Start both API and GUI
                self.start_web_server()
                
                # Small delay to ensure server starts
                await asyncio.sleep(2)
                
                # Start GUI in main thread
                self.start_gui()
                logger.info("✅ Agentic RAG System (Full mode) started successfully!")
            
            else:
                logger.error(f"Unknown mode: {mode}")
                raise ValueError(f"Unknown mode: {mode}")
        
        except Exception as e:
            logger.error(f"❌ System startup failed: {e}")
            self.running = False
            raise
    
    async def test_connections(self):
        """Test all component connections."""
        logger.info("🔧 Testing component connections...")
        
        test_results = {}
        
        # Test Neo4j connection
        try:
            if 'graph_store' in self.components and self.components['graph_store'] is not None:
                # Check if connected
                if hasattr(self.components['graph_store'], 'connected') and self.components['graph_store'].connected:
                    test_results['neo4j'] = "✅ Connected"
                else:
                    test_results['neo4j'] = "❌ Not connected"
            else:
                test_results['neo4j'] = "⚠️ Component not initialized"
        except Exception as e:
            test_results['neo4j'] = f"❌ Failed: {e}"
        
        # Test FAISS index
        try:
            if 'faiss_manager' in self.components and self.components['faiss_manager'] is not None:
                stats = await self.components['faiss_manager'].get_statistics()
                test_results['faiss'] = f"✅ Ready (vectors: {stats.get('total_vectors', 0)})"
            else:
                test_results['faiss'] = "⚠️ Component not initialized"
        except Exception as e:
            test_results['faiss'] = f"❌ Failed: {e}"
        
        # Test embedding service
        try:
            if 'embedding_manager' in self.components and self.components['embedding_manager'] is not None:
                # Test with a small text list
                embeddings = await self.components['embedding_manager'].generate_embeddings(["test"])
                test_results['embeddings'] = f"✅ Connected (dimension: {len(embeddings[0]) if embeddings else 0})"
            else:
                test_results['embeddings'] = "⚠️ Component not initialized"
        except Exception as e:
            test_results['embeddings'] = f"❌ Failed: {e}"
        
        # Log test results
        logger.info("Connection test results:")
        for component, status in test_results.items():
            logger.info(f"  {component}: {status}")
        
        # Check for critical failures
        failed_components = [k for k, v in test_results.items() if v.startswith("❌")]
        if failed_components:
            logger.warning(f"Some components failed connection tests: {failed_components}")
        else:
            logger.info("✅ All connection tests passed!")
    
    async def stop_system(self):
        """Stop the agentic system."""
        logger.info("🛑 Stopping Agentic RAG System...")
        
        self.running = False
        
        # Close connections
        if 'graph_store' in self.components and self.components['graph_store'] is not None:
            try:
                self.components['graph_store'].close()
                logger.info("✓ Graph store connection closed")
            except Exception as e:
                logger.warning(f"Error closing graph store: {e}")
        
        # Save FAISS index
        if 'faiss_manager' in self.components and self.components['faiss_manager'] is not None:
            try:
                self.components['faiss_manager'].save_index()
                logger.info("✓ FAISS index saved")
            except Exception as e:
                logger.warning(f"Error saving FAISS index: {e}")
        
        logger.info("✅ Agentic RAG System stopped successfully")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        return {
            'running': self.running,
            'components': list(self.components.keys()),
            'config_summary': self.config.get_summary(),
            'component_count': len(self.components)
        }


def setup_signal_handlers(manager: AgenticSystemManager):
    """Setup signal handlers for graceful shutdown."""
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, initiating shutdown...")
        asyncio.create_task(manager.stop_system())
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def create_sample_config():
    """Create sample configuration file."""
    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)
    
    sample_config = get_development_config()
    config_file = config_dir / "agentic_rag_config.json"
    
    sample_config.save_to_file(str(config_file))
    logger.info(f"Sample configuration created: {config_file}")


async def main():
    """Main entry point for the Agentic RAG system."""
    parser = argparse.ArgumentParser(description="Complete Agentic Graph RAG System")
    
    parser.add_argument(
        '--mode', 
        choices=['api', 'gui', 'full'], 
        default='full',
        help='Launch mode: api (server only), gui (desktop only), or full (both)'
    )
    
    parser.add_argument(
        '--config', 
        type=str, 
        help='Configuration file path'
    )
    
    parser.add_argument(
        '--env', 
        choices=['development', 'production'], 
        default='development',
        help='Environment configuration'
    )
    
    parser.add_argument(
        '--create-config', 
        action='store_true',
        help='Create sample configuration file and exit'
    )
    
    parser.add_argument(
        '--test-connections', 
        action='store_true',
        help='Test all connections and exit'
    )
    
    parser.add_argument(
        '--log-level', 
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
        default='INFO',
        help='Logging level'
    )
    
    args = parser.parse_args()
    
    # Create sample config if requested
    if args.create_config:
        create_sample_config()
        return
    
    try:
        # Load configuration
        if args.config:
            config = AgenticRAGConfig(args.config)
        elif args.env == 'production':
            config = get_production_config()
        else:
            config = get_development_config()
        
        # Override log level if specified
        config.system.log_level = args.log_level
        config.setup_logging()
        
        logger.info("=" * 60)
        logger.info("🚀 AGENTIC GRAPH RAG SYSTEM")
        logger.info("   Advanced Knowledge Assistant with Multi-Step Reasoning")
        logger.info("=" * 60)
        
        # Display configuration summary
        logger.info("Configuration Summary:")
        for key, value in config.get_summary().items():
            logger.info(f"  {key}: {value}")
        
        # Initialize system manager
        manager = AgenticSystemManager(config)
        
        # Setup signal handlers
        setup_signal_handlers(manager)
        
        # Test connections if requested
        if args.test_connections:
            await manager.initialize_components()
            await manager.test_connections()
            return
        
        # Start system
        await manager.start_system(args.mode)
        
    except KeyboardInterrupt:
        logger.info("🛑 System shutdown requested")
    except Exception as e:
        logger.error(f"❌ System startup failed: {e}")
        logger.error(f"Exception details: {e.__class__.__name__}: {e}")
        sys.exit(1)
    finally:
        if 'manager' in locals():
            await manager.stop_system()


if __name__ == "__main__":
    # Ensure compatibility with both sync and async environments
    try:
        # Check if we're in an existing event loop (Jupyter, etc.)
        try:
            loop = asyncio.get_running_loop()
            # We're in a running loop, need nest_asyncio
            import nest_asyncio
            nest_asyncio.apply()
            loop.create_task(main())
        except RuntimeError:
            # No running loop, safe to create one
            asyncio.run(main())
    except Exception as e:
        # Fallback to simple asyncio.run
        asyncio.run(main())
#!/usr/bin/env python3
"""
Complete Agentic Graph RAG System - Configuration Management
Centralized configuration for all agentic components and system settings.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
import json

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class APIConfig:
    """API configuration settings."""
    # OpenAI/OpenRouter Configuration
    openai_api_key: str = field(default_factory=lambda: os.getenv('OPENAI_API_KEY', ''))
    openrouter_api_key: str = field(default_factory=lambda: os.getenv('OPENROUTER_API_KEY', ''))
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    
    # Model Configuration
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "anthropic/claude-3.5-sonnet"
    max_tokens: int = 4096
    temperature: float = 0.7
    
    # Rate limiting
    requests_per_minute: int = 60
    max_retries: int = 3
    retry_delay: float = 1.0


@dataclass 
class Neo4jConfig:
    """Neo4j database configuration."""
    uri: str = field(default_factory=lambda: os.getenv('NEO4J_URI', 'bolt://localhost:7687'))
    username: str = field(default_factory=lambda: os.getenv('NEO4J_USERNAME', 'neo4j'))
    password: str = field(default_factory=lambda: os.getenv('NEO4J_PASSWORD', 'password'))
    database: str = field(default_factory=lambda: os.getenv('NEO4J_DATABASE', 'neo4j'))
    
    # Connection settings
    max_connection_lifetime: int = 3600
    max_connection_pool_size: int = 50
    connection_timeout: int = 30
    
    # Query settings
    max_query_time: int = 300  # 5 minutes
    batch_size: int = 1000


@dataclass
class FAISSConfig:
    """FAISS index configuration."""
    # Index settings
    index_type: str = "IndexFlatIP"  # Inner product for cosine similarity
    dimension: int = 1536  # text-embedding-3-small dimension
    nlist: int = 100  # Number of clusters for IVF index
    nprobe: int = 10  # Number of clusters to search
    
    # Storage settings
    index_path: str = "data/graphs/faiss"
    save_interval: int = 1000  # Save every N embeddings
    backup_count: int = 5
    
    # Performance settings
    use_gpu: bool = False
    normalize_vectors: bool = True
    batch_size: int = 100


@dataclass
class QueryAnalysisConfig:
    """Query analysis configuration."""
    # Intent classification
    intent_confidence_threshold: float = 0.7
    max_intent_categories: int = 5
    
    # Entity recognition
    entity_confidence_threshold: float = 0.8
    max_entities_per_query: int = 20
    
    # Query decomposition
    max_subqueries: int = 5
    subquery_overlap_threshold: float = 0.3
    
    # Preprocessing
    remove_stop_words: bool = True
    lemmatize: bool = True
    min_query_length: int = 3


@dataclass
class ReasoningConfig:
    """Reasoning engine configuration."""
    # Reasoning strategies
    default_strategy: str = "hybrid"  # forward, backward, hybrid, abductive
    max_reasoning_depth: int = 10
    confidence_threshold: float = 0.6
    
    # Evidence management
    max_evidence_items: int = 50
    evidence_decay_factor: float = 0.9
    min_evidence_support: float = 0.3
    
    # Chain of thought
    enable_cot: bool = True
    cot_steps_limit: int = 10
    cot_confidence_threshold: float = 0.5
    
    # Performance settings
    reasoning_timeout: int = 120  # 2 minutes
    parallel_reasoning: bool = True
    max_parallel_chains: int = 3


@dataclass
class CypherConfig:
    """Cypher query configuration."""
    # Query optimization
    max_query_complexity: int = 1000
    query_timeout: int = 60
    explain_queries: bool = False
    
    # Pattern matching
    max_pattern_length: int = 10
    fuzzy_matching: bool = True
    fuzzy_threshold: float = 0.8
    
    # Traversal settings
    max_hop_distance: int = 6
    default_hop_limit: int = 3
    enable_bidirectional: bool = True
    
    # Caching
    enable_query_cache: bool = True
    cache_size: int = 1000
    cache_ttl: int = 3600  # 1 hour


@dataclass
class GraphVisualizationConfig:
    """Graph visualization configuration."""
    # Layout settings
    default_layout: str = "spring"
    layout_iterations: int = 50
    layout_k: float = 1.0
    
    # Visual settings
    node_size_range: tuple = (10, 50)
    edge_width_range: tuple = (1, 5)
    max_nodes_display: int = 500
    max_edges_display: int = 1000
    
    # Colors
    node_colors: Dict[str, str] = field(default_factory=lambda: {
        "Entity": "#3498db",
        "Concept": "#e74c3c", 
        "Event": "#f39c12",
        "Location": "#27ae60",
        "Person": "#9b59b6",
        "Organization": "#f39c12",
        "Default": "#95a5a6"
    })
    
    # Interactive features
    enable_zoom: bool = True
    enable_pan: bool = True
    enable_selection: bool = True
    enable_hover: bool = True


@dataclass
class WebServerConfig:
    """Web server configuration."""
    # Server settings
    host: str = "localhost"
    port: int = 8000
    reload: bool = False
    workers: int = 1
    
    # CORS settings
    allow_origins: List[str] = field(default_factory=lambda: ["*"])
    allow_methods: List[str] = field(default_factory=lambda: ["GET", "POST", "PUT", "DELETE"])
    allow_headers: List[str] = field(default_factory=lambda: ["*"])
    
    # WebSocket settings
    websocket_ping_interval: int = 30
    websocket_ping_timeout: int = 10
    max_websocket_connections: int = 100
    
    # Security
    enable_https: bool = False
    ssl_cert_path: Optional[str] = None
    ssl_key_path: Optional[str] = None


@dataclass
class GUIConfig:
    """GUI application configuration."""
    # Window settings
    window_title: str = "Agentic RAG System - Interactive Knowledge Assistant"
    window_size: tuple = (1400, 900)
    min_window_size: tuple = (1200, 800)
    
    # Theme settings
    theme: str = "clam"
    font_family: str = "Consolas"
    font_size: int = 10
    
    # Interface settings
    auto_scroll: bool = True
    max_log_entries: int = 1000
    refresh_interval: int = 100  # milliseconds
    
    # Query settings
    default_query_type: str = "factual"
    default_reasoning_depth: int = 3
    default_max_results: int = 10
    stream_results_default: bool = True


@dataclass
class SystemConfig:
    """System-wide configuration."""
    # Paths
    base_path: Path = field(default_factory=lambda: Path(__file__).parent)
    data_path: Path = field(default_factory=lambda: Path(__file__).parent / "data")
    logs_path: Path = field(default_factory=lambda: Path(__file__).parent / "logs")
    temp_path: Path = field(default_factory=lambda: Path(__file__).parent / "temp")
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file: str = "agentic_rag.log"
    max_log_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    
    # Performance
    max_memory_usage: int = 2 * 1024 * 1024 * 1024  # 2GB
    enable_profiling: bool = False
    profiling_output: str = "profile_output.prof"
    
    # Development
    debug_mode: bool = False
    enable_hot_reload: bool = False
    
    def __post_init__(self):
        """Create necessary directories."""
        for path in [self.data_path, self.logs_path, self.temp_path]:
            path.mkdir(exist_ok=True, parents=True)


class AgenticRAGConfig:
    """
    Main configuration class for the Agentic RAG system.
    
    Manages all configuration settings for the complete system.
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize configuration."""
        # Load default configurations
        self.api = APIConfig()
        self.neo4j = Neo4jConfig()
        self.faiss = FAISSConfig()
        self.query_analysis = QueryAnalysisConfig()
        self.reasoning = ReasoningConfig()
        self.cypher = CypherConfig()
        self.visualization = GraphVisualizationConfig()
        self.web_server = WebServerConfig()
        self.gui = GUIConfig()
        self.system = SystemConfig()
        
        # Load from config file if provided
        if config_file:
            self.load_from_file(config_file)
        
        # Validate configuration
        self.validate()
        
        logger.info("Agentic RAG configuration loaded successfully")
    
    def validate(self):
        """Validate configuration settings."""
        issues = []
        
        # API validation
        if not self.api.openai_api_key and not self.api.openrouter_api_key:
            issues.append("No API key configured (OPENAI_API_KEY or OPENROUTER_API_KEY)")
        
        # Neo4j validation
        if not self.neo4j.uri:
            issues.append("Neo4j URI not configured")
        
        # Path validation
        try:
            self.system.data_path.mkdir(exist_ok=True, parents=True)
            self.system.logs_path.mkdir(exist_ok=True, parents=True)
        except Exception as e:
            issues.append(f"Cannot create required directories: {e}")
        
        # FAISS validation
        if self.faiss.dimension <= 0:
            issues.append("FAISS dimension must be positive")
        
        # Performance validation
        if self.reasoning.max_reasoning_depth > 20:
            issues.append("Reasoning depth too high (may cause performance issues)")
        
        if issues:
            logger.warning(f"Configuration validation issues: {issues}")
            for issue in issues:
                logger.warning(f"  - {issue}")
    
    def load_from_file(self, config_file: str):
        """Load configuration from JSON file."""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            # Update configurations from file
            for section, data in config_data.items():
                if hasattr(self, section):
                    config_obj = getattr(self, section)
                    for key, value in data.items():
                        if hasattr(config_obj, key):
                            setattr(config_obj, key, value)
            
            logger.info(f"Configuration loaded from {config_file}")
        
        except Exception as e:
            logger.error(f"Failed to load configuration from {config_file}: {e}")
            raise
    
    def save_to_file(self, config_file: str):
        """Save configuration to JSON file."""
        try:
            config_data = {}
            
            # Export all configurations
            for attr_name in ['api', 'neo4j', 'faiss', 'query_analysis', 
                             'reasoning', 'cypher', 'visualization', 
                             'web_server', 'gui', 'system']:
                if hasattr(self, attr_name):
                    config_obj = getattr(self, attr_name)
                    config_data[attr_name] = {
                        k: v for k, v in config_obj.__dict__.items()
                        if not k.startswith('_') and not callable(v)
                    }
            
            # Convert Path objects to strings
            def convert_paths(obj):
                if isinstance(obj, Path):
                    return str(obj)
                elif isinstance(obj, dict):
                    return {k: convert_paths(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_paths(item) for item in obj]
                return obj
            
            config_data = convert_paths(config_data)
            
            # Save to file
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, default=str)
            
            logger.info(f"Configuration saved to {config_file}")
        
        except Exception as e:
            logger.error(f"Failed to save configuration to {config_file}: {e}")
            raise
    
    def get_component_config(self, component: str) -> Any:
        """Get configuration for a specific component."""
        return getattr(self, component, None)
    
    def update_component_config(self, component: str, config: Dict[str, Any]):
        """Update configuration for a specific component."""
        if hasattr(self, component):
            config_obj = getattr(self, component)
            for key, value in config.items():
                if hasattr(config_obj, key):
                    setattr(config_obj, key, value)
            logger.debug(f"Updated {component} configuration")
        else:
            logger.error(f"Unknown component: {component}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get configuration summary."""
        return {
            'api_configured': bool(self.api.openai_api_key or self.api.openrouter_api_key),
            'neo4j_configured': bool(self.neo4j.uri and self.neo4j.username),
            'faiss_dimension': self.faiss.dimension,
            'reasoning_depth': self.reasoning.max_reasoning_depth,
            'max_query_time': self.cypher.query_timeout,
            'server_port': self.web_server.port,
            'debug_mode': self.system.debug_mode,
            'log_level': self.system.log_level
        }
    
    def setup_logging(self):
        """Setup logging based on configuration."""
        log_file = self.system.logs_path / self.system.log_file
        
        logging.basicConfig(
            level=getattr(logging, self.system.log_level.upper()),
            format=self.system.log_format,
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        
        logger.info(f"Logging setup complete (level: {self.system.log_level})")


# Global configuration instance
config = AgenticRAGConfig()


# Convenience functions
def get_config() -> AgenticRAGConfig:
    """Get the global configuration instance."""
    return config


def load_config(config_file: str) -> AgenticRAGConfig:
    """Load configuration from file."""
    global config
    config = AgenticRAGConfig(config_file)
    return config


def save_config(config_file: str):
    """Save configuration to file."""
    config.save_to_file(config_file)


# Environment-specific configurations
def get_development_config() -> AgenticRAGConfig:
    """Get development configuration."""
    dev_config = AgenticRAGConfig()
    dev_config.system.debug_mode = True
    dev_config.system.log_level = "DEBUG"
    dev_config.web_server.reload = True
    dev_config.reasoning.enable_cot = True
    dev_config.cypher.explain_queries = True
    return dev_config


def get_production_config() -> AgenticRAGConfig:
    """Get production configuration."""
    prod_config = AgenticRAGConfig()
    prod_config.system.debug_mode = False
    prod_config.system.log_level = "INFO"
    prod_config.web_server.reload = False
    prod_config.web_server.workers = 4
    prod_config.faiss.batch_size = 500
    prod_config.neo4j.max_connection_pool_size = 100
    return prod_config


if __name__ == "__main__":
    # Example usage
    config = AgenticRAGConfig()
    config.setup_logging()
    
    print("Agentic RAG Configuration Summary:")
    print(json.dumps(config.get_summary(), indent=2))
    
    # Save example configuration
    config.save_to_file("config/agentic_rag_config.json")
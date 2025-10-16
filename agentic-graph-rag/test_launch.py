#!/usr/bin/env python3
"""
Test launch script to debug the agentic system startup.
"""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

print("Starting test launch...")
print("Python version:", sys.version)
print("Current directory:", os.getcwd())

try:
    print("Testing basic imports...")
    import asyncio
    print("✓ asyncio imported")
    
    import logging
    print("✓ logging imported")
    
    from config import AgenticRAGConfig, get_development_config
    print("✓ config imported")
    
    # Test FAISS import (this might be slow)
    print("Testing FAISS import (may take a moment)...")
    from core.faiss_index_manager import FAISSIndexManager
    print("✓ FAISS imported successfully")
    
    from core.neo4j_graph_store_v2 import Neo4jGraphStore
    print("✓ Neo4j imported")
    
    from core.embedding_manager_v2 import EmbeddingManager
    print("✓ Embedding manager imported")
    
    print("All imports successful!")
    
    # Try basic config
    config = get_development_config()
    print("✓ Configuration loaded")
    
    print("Basic test completed successfully!")
    
except Exception as e:
    print(f"❌ Error during test: {e}")
    import traceback
    traceback.print_exc()
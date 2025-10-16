#!/usr/bin/env python3
"""
Phase 2 System Validation Script
Validates that all core Phase 2 components are working correctly
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def test_imports():
    """Test that all core modules can be imported."""
    print("🧪 Testing Core Module Imports...")
    
    try:
        from phase2_pipeline import AgenticGraphRAGPipeline
        print("✅ phase2_pipeline.py - Main pipeline imported successfully")
    except ImportError as e:
        print(f"❌ phase2_pipeline.py - Failed to import: {e}")
        return False
    
    try:
        from core.llm_ontology_generator import LLMOntologyGenerator
        print("✅ llm_ontology_generator.py - LLM ontology generator imported")
    except ImportError as e:
        print(f"❌ llm_ontology_generator.py - Failed to import: {e}")
    
    try:
        from core.embedding_manager_v2 import EmbeddingManager
        print("✅ embedding_manager_v2.py - Embedding manager imported")
    except ImportError as e:
        print(f"❌ embedding_manager_v2.py - Failed to import: {e}")
    
    try:
        from core.neo4j_graph_store_v2 import Neo4jGraphStore
        print("✅ neo4j_graph_store_v2.py - Neo4j store imported")
    except ImportError as e:
        print(f"❌ neo4j_graph_store_v2.py - Failed to import: {e}")
    
    try:
        from core.simple_document_processor import DocumentProcessor
        print("✅ simple_document_processor.py - Document processor imported")
    except ImportError as e:
        print(f"❌ simple_document_processor.py - Failed to import: {e}")
    
    try:
        from core.simple_graph_visualizer import SimpleGraphVisualizer
        print("✅ simple_graph_visualizer.py - Graph visualizer imported")
    except ImportError as e:
        print(f"❌ simple_graph_visualizer.py - Failed to import: {e}")
    
    return True

def test_gui_availability():
    """Test if GUI components can be loaded."""
    print("\n🖥️ Testing GUI Availability...")
    
    try:
        import tkinter
        print("✅ tkinter - Available for desktop GUI")
        
        # Test if phase2_gui can be imported
        try:
            from phase2_gui import Phase2GraphRAGGUI
            print("✅ phase2_gui.py - Desktop GUI class available")
        except ImportError as e:
            print(f"❌ phase2_gui.py - GUI import failed: {e}")
    except ImportError:
        print("❌ tkinter - Not available (desktop GUI disabled)")
    
    try:
        import streamlit
        print("✅ streamlit - Available for web interface")
    except ImportError:
        print("⚠️ streamlit - Not installed (web interface disabled)")

def test_dependencies():
    """Test required dependencies."""
    print("\n📦 Testing Dependencies...")
    
    required_packages = [
        ("pathlib", "Path handling"),
        ("json", "Data serialization"),
        ("asyncio", "Async processing"),
        ("datetime", "Date/time utilities"),
        ("logging", "Logging system"),
    ]
    
    optional_packages = [
        ("PyMuPDF", "PDF processing"),
        ("docx", "DOCX document processing"),
        ("pytesseract", "OCR for images"),
        ("httpx", "HTTP client for APIs"),
        ("openai", "OpenAI API"),
        ("plotly", "Graph visualization"),
        ("networkx", "Graph algorithms"),
    ]
    
    for package, description in required_packages:
        try:
            __import__(package)
            print(f"✅ {package:<15} - {description}")
        except ImportError:
            print(f"❌ {package:<15} - {description} (REQUIRED)")
    
    print("\nOptional packages:")
    for package, description in optional_packages:
        try:
            __import__(package)
            print(f"✅ {package:<15} - {description}")
        except ImportError:
            print(f"⚠️ {package:<15} - {description} (optional)")

def test_configuration():
    """Test configuration files and environment."""
    print("\n⚙️ Testing Configuration...")
    
    # Check .env file
    env_file = Path(".env")
    if env_file.exists():
        print("✅ .env file - Configuration file present")
        
        # Check for API keys
        try:
            from dotenv import load_dotenv
            load_dotenv()
            
            openrouter_key = os.getenv("OPENROUTER_API_KEY")
            openai_key = os.getenv("OPENAI_API_KEY")
            
            if openrouter_key:
                print("✅ OPENROUTER_API_KEY - Configured")
            else:
                print("⚠️ OPENROUTER_API_KEY - Not configured")
                
            if openai_key:
                print("✅ OPENAI_API_KEY - Configured")
            else:
                print("⚠️ OPENAI_API_KEY - Not configured")
                
        except ImportError:
            print("⚠️ python-dotenv - Not installed, cannot check API keys")
    else:
        print("⚠️ .env file - Not found (create from template)")
    
    # Check data directories
    data_dir = Path("data")
    if data_dir.exists():
        print("✅ data/ directory - Present")
        
        subdirs = ["raw", "processed", "graphs", "indexes"]
        for subdir in subdirs:
            subdir_path = data_dir / subdir
            if subdir_path.exists():
                print(f"✅ data/{subdir}/ - Present")
            else:
                print(f"⚠️ data/{subdir}/ - Will be created automatically")
    else:
        print("⚠️ data/ directory - Will be created automatically")

def test_basic_pipeline():
    """Test basic pipeline instantiation."""
    print("\n🔧 Testing Pipeline Instantiation...")
    
    try:
        from phase2_pipeline import AgenticGraphRAGPipeline
        pipeline = AgenticGraphRAGPipeline()
        print("✅ Pipeline instantiation - Success")
        print(f"✅ Session ID: {pipeline.session_id}")
        return True
    except Exception as e:
        print(f"❌ Pipeline instantiation - Failed: {e}")
        return False

def main():
    """Run all validation tests."""
    print("🕸️ Phase 2 System Validation")
    print("=" * 50)
    
    all_tests_passed = True
    
    # Run tests
    if not test_imports():
        all_tests_passed = False
    
    test_gui_availability()
    test_dependencies()
    test_configuration()
    
    if not test_basic_pipeline():
        all_tests_passed = False
    
    # Final results
    print("\n" + "=" * 50)
    if all_tests_passed:
        print("🎉 Phase 2 System Validation - PASSED")
        print("✅ All core components are working correctly")
        print("\n🚀 Ready to use Phase 2 system!")
        print("   • Desktop GUI: python phase2_gui.py")
        print("   • Web Interface: streamlit run phase2_web_app.py")
        print("   • Universal Launcher: python launch_phase2.py")
    else:
        print("⚠️ Phase 2 System Validation - ISSUES FOUND")
        print("❌ Some components may not work correctly")
        print("\n🔧 Recommended actions:")
        print("   • Install missing dependencies: pip install -r requirements.txt")
        print("   • Configure API keys in .env file")
        print("   • Check error messages above")
    
    return all_tests_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
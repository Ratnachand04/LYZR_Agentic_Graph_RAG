#!/usr/bin/env python3
"""
Agentic Graph RAG Pipeline - Phase 2 Launcher
Choose your preferred interface: GUI, Web App, or Command Line
"""

import sys
import os
import subprocess
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))


def check_dependencies():
    """Check if required dependencies are available."""
    issues = []
    
    # Check tkinter for GUI
    try:
        import tkinter
        have_tkinter = True
    except ImportError:
        have_tkinter = False
        issues.append("tkinter not available (GUI interface disabled)")
    
    # Check streamlit for web app
    try:
        import streamlit
        have_streamlit = True
    except ImportError:
        have_streamlit = False
        issues.append("streamlit not available (Web interface disabled)")
    
    # Check core pipeline
    try:
        from phase2_pipeline import AgenticGraphRAGPipeline
        have_pipeline = True
    except ImportError as e:
        have_pipeline = False
        issues.append(f"Phase 2 pipeline not available: {e}")
    
    return {
        'tkinter': have_tkinter,
        'streamlit': have_streamlit,
        'pipeline': have_pipeline,
        'issues': issues
    }


def show_banner():
    """Display application banner."""
    banner = """
╭─────────────────────────────────────────────────────────────╮
│                                                             │
│    🕸️  Agentic Graph RAG Pipeline - Phase 2               │
│                                                             │
│    Advanced Document-to-Graph Processing Platform          │
│    • LLM Ontology Generation                                │
│    • Vector Embeddings & Similarity Search                 │
│    • Neo4j Graph Database Integration                       │
│    • Interactive 3D Visualization                          │
│                                                             │
╰─────────────────────────────────────────────────────────────╯
"""
    print(banner)


def show_menu(deps):
    """Display interface selection menu."""
    print("\nAvailable Interfaces:")
    print("=" * 50)
    
    if deps['tkinter'] and deps['pipeline']:
        print("1. 🖥️  Desktop GUI (Tkinter)")
        print("   - Native desktop application")
        print("   - Full-featured interface")
        print("   - Offline processing")
    else:
        print("1. ❌ Desktop GUI (Not Available)")
        if not deps['tkinter']:
            print("   - tkinter not installed")
        if not deps['pipeline']:
            print("   - Pipeline not available")
    
    print()
    
    if deps['streamlit'] and deps['pipeline']:
        print("2. 🌐 Web Interface (Streamlit)")
        print("   - Modern web application")
        print("   - Real-time updates")
        print("   - Cross-platform browser access")
    else:
        print("2. ❌ Web Interface (Not Available)")
        if not deps['streamlit']:
            print("   - streamlit not installed")
        if not deps['pipeline']:
            print("   - Pipeline not available")
    
    print()
    
    if deps['pipeline']:
        print("3. ⚡ Command Line Interface")
        print("   - Direct pipeline execution")
        print("   - Scriptable automation")
        print("   - Minimal dependencies")
    else:
        print("3. ❌ Command Line (Not Available)")
        print("   - Pipeline not available")
    
    print()
    print("4. 🔧 System Check & Setup")
    print("5. 📚 Documentation")
    print("0. ❌ Exit")
    
    if deps['issues']:
        print("\n⚠️  Issues Detected:")
        for issue in deps['issues']:
            print(f"   • {issue}")


def launch_gui():
    """Launch the desktop GUI interface."""
    try:
        from phase2_gui import Phase2GraphRAGGUI
        print("\n🚀 Starting Desktop GUI...")
        app = Phase2GraphRAGGUI()
        app.run()
    except Exception as e:
        print(f"❌ Failed to launch GUI: {e}")
        input("Press Enter to continue...")


def launch_web_app():
    """Launch the web interface."""
    try:
        print("\n🚀 Starting Web Interface...")
        print("📍 Opening http://localhost:8501")
        print("⏹️  Press Ctrl+C to stop")
        
        # Launch streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "phase2_web_app.py",
            "--server.port", "8501",
            "--server.headless", "false"
        ])
    except KeyboardInterrupt:
        print("\n🛑 Web interface stopped")
    except Exception as e:
        print(f"❌ Failed to launch web interface: {e}")
        input("Press Enter to continue...")


def launch_cli():
    """Launch command line interface."""
    try:
        from phase2_pipeline import AgenticGraphRAGPipeline
        
        print("\n⚡ Command Line Interface")
        print("=" * 40)
        
        # Simple CLI workflow
        pipeline = AgenticGraphRAGPipeline()
        
        print("\n📁 Enter document paths (comma-separated):")
        input_paths = input("> ").strip().split(",")
        input_paths = [p.strip() for p in input_paths if p.strip()]
        
        if not input_paths:
            print("❌ No input paths provided")
            return
        
        print("\n📄 Output filename:")
        output_path = input("> ").strip() or "agentic_graph_output.html"
        
        print("\n🧠 Enable LLM ontology generation? (y/n):")
        enable_llm = input("> ").lower().startswith('y')
        
        print("\n🔢 Enable embeddings? (y/n):")
        enable_embeddings = input("> ").lower().startswith('y')
        
        print("\n🗄️ Store in Neo4j? (y/n):")
        enable_neo4j = input("> ").lower().startswith('y')
        
        print(f"\n🚀 Processing {len(input_paths)} documents...")
        
        # Run pipeline
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        results = loop.run_until_complete(
            pipeline.run_full_pipeline(
                input_paths=input_paths,
                output_path=output_path,
                enable_llm=enable_llm,
                enable_embeddings=enable_embeddings,
                enable_neo4j=enable_neo4j
            )
        )
        
        print(f"\n✅ Processing completed!")
        print(f"📊 Session ID: {results['session_id']}")
        print(f"📄 Documents: {results['statistics']['documents_processed']}")
        print(f"🔖 Entities: {results['statistics']['entities_extracted']}")
        print(f"🔗 Relations: {results['statistics']['relationships_extracted']}")
        print(f"🌐 Output: {results['outputs']['visualization_path']}")
        
    except Exception as e:
        print(f"❌ CLI execution failed: {e}")
    
    input("\nPress Enter to continue...")


def show_system_check():
    """Show detailed system check and setup information."""
    print("\n🔧 System Check & Setup")
    print("=" * 50)
    
    deps = check_dependencies()
    
    # Python version
    print(f"🐍 Python: {sys.version}")
    
    # Dependencies
    print("\n📦 Dependencies:")
    packages = [
        ("tkinter", "Desktop GUI support"),
        ("streamlit", "Web interface"),
        ("asyncio", "Async processing"),
        ("pathlib", "Path handling"),
        ("json", "Data serialization")
    ]
    
    for package, description in packages:
        try:
            __import__(package)
            status = "✅"
        except ImportError:
            status = "❌"
        print(f"   {status} {package:<15} - {description}")
    
    # Phase 2 components
    print("\n🛠️ Phase 2 Components:")
    components = [
        ("phase2_pipeline.py", "Main processing pipeline"),
        ("neo4j_graph_store_v2.py", "Neo4j integration"),
        ("llm_ontology_generator.py", "LLM ontology generation"),
        ("embedding_manager_v2.py", "Embedding management"),
        ("simple_graph_visualizer.py", "Graph visualization")
    ]
    
    for component, description in components:
        if Path(component).exists():
            status = "✅"
        else:
            status = "❌"
        print(f"   {status} {component:<25} - {description}")
    
    # Configuration
    print("\n⚙️ Configuration:")
    config_items = [
        ("OPENROUTER_API_KEY", "LLM ontology generation"),
        ("OPENAI_API_KEY", "Embedding generation"),
        (".env file", "Environment configuration")
    ]
    
    for item, description in config_items:
        if item.startswith('.'):
            # File check
            status = "✅" if Path(item).exists() else "⚠️"
        else:
            # Environment variable check
            status = "✅" if os.getenv(item) else "⚠️"
        print(f"   {status} {item:<20} - {description}")
    
    # Installation help
    print("\n📋 Quick Setup:")
    print("   1. Install dependencies: pip install -r requirements.txt")
    print("   2. Copy .env.phase2.example to .env")
    print("   3. Add your API keys to .env file")
    print("   4. Run validation: python test_phase2.py")
    
    input("\nPress Enter to continue...")


def show_documentation():
    """Show documentation and help."""
    print("\n📚 Documentation")
    print("=" * 50)
    
    docs = """
🕸️ Agentic Graph RAG Pipeline - Phase 2

OVERVIEW:
Production-grade document processing system that transforms unstructured 
documents into interactive knowledge graphs using LLM-powered ontology 
generation and vector embeddings.

KEY FEATURES:
• Multi-format document processing (PDF, DOCX, TXT, MD, Images)
• LLM-powered entity and relationship extraction
• Vector embeddings with similarity search
• Neo4j graph database integration
• Interactive 3D graph visualization
• Real-time processing with progress tracking

PROCESSING PIPELINE:
1. Document Ingestion - Multi-format text extraction
2. LLM Ontology Generation - Structured entity/relationship extraction
3. Embedding Generation - Vector representations for similarity
4. Entity Resolution - Deduplication and merging
5. Graph Storage - Neo4j database integration
6. Visualization - Interactive 3D graph rendering

CONFIGURATION:
• OpenRouter API Key - For LLM ontology generation
• OpenAI API Key - For embedding generation
• Neo4j Database - Optional graph storage
• Custom models and parameters

FILES:
• phase2_pipeline.py - Main processing pipeline
• phase2_gui.py - Desktop GUI interface  
• phase2_web_app.py - Web interface
• test_phase2.py - System validation
• .env.phase2.example - Configuration template

USAGE EXAMPLES:
1. Desktop GUI: python phase2_gui.py
2. Web Interface: streamlit run phase2_web_app.py
3. Command Line: python phase2_pipeline.py
4. Validation: python test_phase2.py

TROUBLESHOOTING:
• Check API keys are set correctly
• Verify all dependencies installed
• Run test_phase2.py for validation
• Check processing logs for errors

For detailed help, see the README.md file or run the validation script.
"""
    
    print(docs)
    input("Press Enter to continue...")


def main():
    """Main launcher application."""
    try:
        while True:
            # Clear screen (platform independent)
            os.system('cls' if os.name == 'nt' else 'clear')
            
            show_banner()
            deps = check_dependencies()
            show_menu(deps)
            
            try:
                choice = input("\n📍 Select an option (0-5): ").strip()
                
                if choice == '0':
                    print("\n👋 Goodbye!")
                    break
                
                elif choice == '1':
                    if deps['tkinter'] and deps['pipeline']:
                        launch_gui()
                    else:
                        print("\n❌ Desktop GUI not available")
                        input("Press Enter to continue...")
                
                elif choice == '2':
                    if deps['streamlit'] and deps['pipeline']:
                        launch_web_app()
                    else:
                        print("\n❌ Web interface not available")
                        input("Press Enter to continue...")
                
                elif choice == '3':
                    if deps['pipeline']:
                        launch_cli()
                    else:
                        print("\n❌ Command line interface not available")
                        input("Press Enter to continue...")
                
                elif choice == '4':
                    show_system_check()
                
                elif choice == '5':
                    show_documentation()
                
                else:
                    print(f"\n❌ Invalid choice: {choice}")
                    input("Press Enter to continue...")
            
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                input("Press Enter to continue...")
    
    except Exception as e:
        print(f"\nFatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
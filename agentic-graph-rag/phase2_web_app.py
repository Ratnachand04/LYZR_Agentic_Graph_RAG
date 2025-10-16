#!/usr/bin/env python3
"""
Phase 2 Web Interface for Agentic Graph RAG Pipeline
Modern web application using Streamlit for document-to-graph processing.
"""

import streamlit as st
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional
import tempfile
import time
import webbrowser
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import Phase 2 components
try:
    from phase2_pipeline import AgenticGraphRAGPipeline
    HAVE_PIPELINE = True
except ImportError as e:
    HAVE_PIPELINE = False
    PIPELINE_ERROR = str(e)


# Page configuration
st.set_page_config(
    page_title="Agentic Graph RAG Pipeline - Phase 2",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    padding: 1rem 0;
    border-bottom: 2px solid #e0e0e0;
    margin-bottom: 2rem;
}
.status-success {
    color: #4CAF50;
    font-weight: bold;
}
.status-warning {
    color: #FF9800;
    font-weight: bold;
}
.status-error {
    color: #F44336;
    font-weight: bold;
}
.metrics-container {
    background: #f8f9fa;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 1rem 0;
}
.processing-log {
    background: #1e1e1e;
    color: #ffffff;
    padding: 1rem;
    border-radius: 0.5rem;
    font-family: 'Courier New', monospace;
    font-size: 0.8rem;
    max-height: 300px;
    overflow-y: auto;
}
</style>
""", unsafe_allow_html=True)


class Phase2WebApp:
    """Streamlit web application for Agentic Graph RAG Pipeline."""
    
    def __init__(self):
        """Initialize the web application."""
        self.pipeline = None
        if HAVE_PIPELINE:
            try:
                self.pipeline = AgenticGraphRAGPipeline()
            except Exception as e:
                st.error(f"Failed to initialize pipeline: {e}")
        
        # Initialize session state
        if 'processing_results' not in st.session_state:
            st.session_state.processing_results = None
        if 'uploaded_files' not in st.session_state:
            st.session_state.uploaded_files = []
        if 'processing_log' not in st.session_state:
            st.session_state.processing_log = []
        if 'session_id' not in st.session_state:
            st.session_state.session_id = None
    
    def log_message(self, message: str, level: str = "info"):
        """Add message to processing log."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = {
            "timestamp": timestamp,
            "level": level,
            "message": message
        }
        st.session_state.processing_log.append(log_entry)
        
        # Keep only last 100 messages
        if len(st.session_state.processing_log) > 100:
            st.session_state.processing_log = st.session_state.processing_log[-100:]
    
    def render_header(self):
        """Render the main header."""
        st.markdown('<div class="main-header">', unsafe_allow_html=True)
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.title("🕸️ Agentic Graph RAG Pipeline")
            st.markdown("**Phase 2**: Advanced Document-to-Graph Processing with LLM Ontology Generation")
        
        with col2:
            if st.session_state.processing_results:
                st.success(f"Session: {st.session_state.session_id[:8]}...")
            else:
                st.info("Ready to Process")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render the sidebar configuration."""
        st.sidebar.title("⚙️ Configuration")
        
        # API Configuration
        with st.sidebar.expander("🔑 API Keys", expanded=True):
            openrouter_key = st.text_input(
                "OpenRouter API Key",
                value=os.getenv('OPENROUTER_API_KEY', ''),
                type="password",
                help="Required for LLM ontology generation"
            )
            
            openai_key = st.text_input(
                "OpenAI API Key", 
                value=os.getenv('OPENAI_API_KEY', ''),
                type="password",
                help="Required for embeddings generation"
            )
            
            if st.button("💾 Save Configuration"):
                os.environ['OPENROUTER_API_KEY'] = openrouter_key
                os.environ['OPENAI_API_KEY'] = openai_key
                
                # Save to .env file
                env_content = f"""OPENROUTER_API_KEY={openrouter_key}
OPENAI_API_KEY={openai_key}
"""
                try:
                    with open('.env', 'w') as f:
                        f.write(env_content)
                    st.sidebar.success("Configuration saved!")
                except Exception as e:
                    st.sidebar.error(f"Save failed: {e}")
        
        # Model Configuration
        with st.sidebar.expander("🤖 Model Settings"):
            llm_model = st.selectbox(
                "LLM Model",
                ["microsoft/wizardlm-2-8x22b", "anthropic/claude-3-sonnet", 
                 "openai/gpt-4-turbo", "google/gemini-pro"],
                help="Model for ontology generation"
            )
            
            embedding_model = st.selectbox(
                "Embedding Model",
                ["text-embedding-ada-002", "text-embedding-3-small", "text-embedding-3-large"],
                help="Model for generating embeddings"
            )
        
        # Component Status
        with st.sidebar.expander("📊 System Status"):
            self.render_component_status()
        
        return {
            'openrouter_key': openrouter_key,
            'openai_key': openai_key,
            'llm_model': llm_model,
            'embedding_model': embedding_model
        }
    
    def render_component_status(self):
        """Render system component status."""
        components = [
            ("Pipeline", "✅" if HAVE_PIPELINE else "❌", "Core pipeline components"),
            ("LLM", "✅" if os.getenv('OPENROUTER_API_KEY') else "⚠️", "Ontology generation"),
            ("Embeddings", "✅" if os.getenv('OPENAI_API_KEY') else "⚠️", "Vector embeddings"),
            ("Neo4j", "⚠️", "Graph database (optional)"),
        ]
        
        for name, status, description in components:
            col1, col2, col3 = st.columns([2, 1, 3])
            with col1:
                st.text(name)
            with col2:
                st.text(status)
            with col3:
                st.caption(description)
    
    def render_file_upload(self):
        """Render file upload interface."""
        st.header("📁 Document Selection")
        
        uploaded_files = st.file_uploader(
            "Upload documents to process",
            type=['pdf', 'docx', 'txt', 'md', 'jpg', 'png', 'jpeg'],
            accept_multiple_files=True,
            help="Supported formats: PDF, DOCX, TXT, MD, JPG, PNG"
        )
        
        if uploaded_files:
            st.session_state.uploaded_files = uploaded_files
            
            with st.expander(f"📋 Selected Files ({len(uploaded_files)})"):
                for i, file in enumerate(uploaded_files):
                    col1, col2, col3 = st.columns([3, 1, 1])
                    with col1:
                        st.text(file.name)
                    with col2:
                        st.text(f"{file.size / 1024:.1f} KB")
                    with col3:
                        st.text(file.type or "unknown")
        
        return uploaded_files
    
    def render_processing_options(self):
        """Render processing options."""
        st.header("⚙️ Processing Options")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            enable_llm = st.checkbox(
                "🧠 LLM Ontology Generation",
                value=True,
                help="Extract entities and relationships using LLM"
            )
        
        with col2:
            enable_embeddings = st.checkbox(
                "🔢 Generate Embeddings",
                value=True,
                help="Create vector embeddings for similarity search"
            )
        
        with col3:
            enable_neo4j = st.checkbox(
                "🗄️ Store in Neo4j",
                value=False,
                help="Save results to Neo4j graph database"
            )
        
        # Output configuration
        output_filename = st.text_input(
            "Output Filename",
            value="agentic_graph_output.html",
            help="Name for the generated visualization file"
        )
        
        return {
            'enable_llm': enable_llm,
            'enable_embeddings': enable_embeddings,
            'enable_neo4j': enable_neo4j,
            'output_filename': output_filename
        }
    
    def render_processing_controls(self, config: Dict, options: Dict, uploaded_files):
        """Render processing controls and execution."""
        st.header("🚀 Processing")
        
        # Validation
        can_process = True
        issues = []
        
        if not uploaded_files:
            issues.append("No files uploaded")
            can_process = False
        
        if options['enable_llm'] and not config['openrouter_key']:
            issues.append("LLM ontology enabled but OpenRouter API key missing")
        
        if options['enable_embeddings'] and not config['openai_key']:
            issues.append("Embeddings enabled but OpenAI API key missing")
        
        if not HAVE_PIPELINE:
            issues.append(f"Pipeline not available: {PIPELINE_ERROR}")
            can_process = False
        
        # Display issues
        if issues:
            for issue in issues:
                st.warning(f"⚠️ {issue}")
        
        # Processing button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button(
                "🎯 Start Processing",
                disabled=not can_process,
                help="Begin document-to-graph processing pipeline",
                type="primary"
            ):
                if can_process:
                    self.run_processing(config, options, uploaded_files)
        
        # Results section
        if st.session_state.processing_results:
            self.render_results()
    
    def run_processing(self, config: Dict, options: Dict, uploaded_files):
        """Execute the processing pipeline."""
        if not self.pipeline:
            st.error("Pipeline not available")
            return
        
        # Set environment variables
        os.environ['OPENROUTER_API_KEY'] = config['openrouter_key']
        os.environ['OPENAI_API_KEY'] = config['openai_key']
        
        # Save uploaded files temporarily
        temp_files = []
        try:
            with st.spinner("💾 Saving uploaded files..."):
                for uploaded_file in uploaded_files:
                    temp_file = tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=f"_{uploaded_file.name}"
                    )
                    temp_file.write(uploaded_file.read())
                    temp_file.close()
                    temp_files.append(temp_file.name)
                    self.log_message(f"Saved: {uploaded_file.name}")
            
            # Run pipeline
            with st.spinner("🔄 Processing documents..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Create event loop for async processing
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                try:
                    # Update progress during processing
                    status_text.text("Starting pipeline...")
                    progress_bar.progress(10)
                    
                    results = loop.run_until_complete(
                        self.pipeline.run_full_pipeline(
                            input_paths=temp_files,
                            output_path=options['output_filename'],
                            enable_llm=options['enable_llm'],
                            enable_embeddings=options['enable_embeddings'],
                            enable_neo4j=options['enable_neo4j']
                        )
                    )
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Processing completed!")
                    
                    # Store results
                    st.session_state.processing_results = results
                    st.session_state.session_id = results.get('session_id', 'unknown')
                    
                    self.log_message("Processing completed successfully!", "success")
                    
                    # Show success message
                    st.success(f"🎉 Processing completed! Session ID: {st.session_state.session_id}")
                    
                    # Auto-rerun to show results
                    time.sleep(1)
                    st.rerun()
                
                finally:
                    loop.close()
        
        except Exception as e:
            st.error(f"❌ Processing failed: {e}")
            self.log_message(f"Processing failed: {e}", "error")
        
        finally:
            # Clean up temporary files
            for temp_file in temp_files:
                try:
                    os.unlink(temp_file)
                except:
                    pass
    
    def render_results(self):
        """Render processing results."""
        st.header("📊 Results")
        
        results = st.session_state.processing_results
        if not results:
            st.info("No results available. Run processing first.")
            return
        
        # Results overview
        col1, col2, col3, col4 = st.columns(4)
        
        stats = results.get('statistics', {})
        with col1:
            st.metric("📄 Documents", stats.get('documents_processed', 0))
        with col2:
            st.metric("🔖 Entities", stats.get('entities_extracted', 0))
        with col3:
            st.metric("🔗 Relations", stats.get('relationships_extracted', 0))
        with col4:
            st.metric("⏱️ Duration", results.get('total_duration', 'N/A'))
        
        # Action buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🌐 Open Visualization", type="primary"):
                output_file = results.get('outputs', {}).get('visualization_path')
                if output_file and Path(output_file).exists():
                    webbrowser.open(f"file://{Path(output_file).absolute()}")
                    st.success("Opening visualization in browser...")
                else:
                    st.error("Visualization file not found")
        
        with col2:
            if st.button("💾 Download Results"):
                try:
                    json_str = json.dumps(results, indent=2, default=str)
                    st.download_button(
                        label="📥 Download JSON",
                        data=json_str,
                        file_name=f"results_{st.session_state.session_id}.json",
                        mime="application/json"
                    )
                except Exception as e:
                    st.error(f"Export failed: {e}")
        
        with col3:
            if st.button("🗄️ Open Neo4j Browser"):
                webbrowser.open("http://localhost:7474")
                st.info("Opening Neo4j Browser...")
        
        # Detailed results
        with st.expander("📋 Detailed Results"):
            st.json(results)
    
    def render_processing_log(self):
        """Render processing log."""
        if st.session_state.processing_log:
            st.header("📝 Processing Log")
            
            # Format log messages
            log_html = "<div class='processing-log'>"
            for entry in reversed(st.session_state.processing_log[-20:]):  # Show last 20 messages
                level_color = {
                    'info': '#ffffff',
                    'success': '#4CAF50',
                    'warning': '#FF9800',
                    'error': '#F44336'
                }.get(entry['level'], '#ffffff')
                
                log_html += f"<div style='color: {level_color}'>[{entry['timestamp']}] {entry['message']}</div>"
            
            log_html += "</div>"
            st.markdown(log_html, unsafe_allow_html=True)
            
            if st.button("🗑️ Clear Log"):
                st.session_state.processing_log = []
                st.rerun()
    
    def run(self):
        """Run the web application."""
        self.render_header()
        
        # Sidebar configuration
        config = self.render_sidebar()
        
        # Main content
        uploaded_files = self.render_file_upload()
        
        options = self.render_processing_options()
        
        self.render_processing_controls(config, options, uploaded_files)
        
        self.render_processing_log()


def main():
    """Main entry point."""
    app = Phase2WebApp()
    app.run()


if __name__ == "__main__":
    main()
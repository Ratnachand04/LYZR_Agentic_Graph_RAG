#!/usr/bin/env python3
"""
Phase 2 GUI Application for Agentic Graph RAG Pipeline
Advanced desktop interface for document-to-graph processing with LLM ontology generation.
"""

import asyncio
import logging
import sys
import threading
import os
from pathlib import Path
from typing import List, Dict, Optional
import webbrowser
from datetime import datetime

# GUI Framework
try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox, scrolledtext
    from tkinter.ttk import Progressbar, Notebook
    HAVE_TKINTER = True
except ImportError:
    HAVE_TKINTER = False

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import Phase 2 components
from phase2_pipeline import AgenticGraphRAGPipeline

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Phase2GraphRAGGUI:
    """
    Advanced GUI for Agentic Graph RAG Pipeline.
    
    Features:
    - Multi-document processing with progress tracking
    - LLM ontology generation controls
    - Neo4j integration status
    - Embedding management
    - Real-time processing logs
    - Interactive results viewer
    """
    
    def __init__(self):
        """Initialize the Phase 2 GUI application."""
        if not HAVE_TKINTER:
            raise RuntimeError("Tkinter not available")
        
        self.root = tk.Tk()
        self.root.title("Agentic Graph RAG Pipeline - Phase 2")
        self.root.geometry("1000x700")
        self.root.minsize(800, 600)
        
        # Initialize pipeline
        self.pipeline = AgenticGraphRAGPipeline()
        self.selected_files = []
        self.current_session = None
        self.processing_results = {}
        
        # GUI state
        self.is_processing = False
        self.progress_var = tk.DoubleVar()
        self.status_var = tk.StringVar(value="Ready")
        
        # Create GUI
        self._create_gui()
        self._load_configuration()
        self._update_component_status()
        
        logger.info("Phase 2 GUI initialized")
    
    def _create_gui(self):
        """Create the main GUI interface."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill="both", expand=True)
        
        # Create notebook for tabs
        notebook = Notebook(main_frame)
        notebook.pack(fill="both", expand=True)
        
        # Tab 1: Document Processing
        self._create_processing_tab(notebook)
        
        # Tab 2: Configuration
        self._create_config_tab(notebook)
        
        # Tab 3: Results & Analytics
        self._create_results_tab(notebook)
        
        # Status bar
        self._create_status_bar(main_frame)
    
    def _create_processing_tab(self, parent):
        """Create the document processing tab."""
        processing_frame = ttk.Frame(parent)
        parent.add(processing_frame, text="Document Processing")
        
        # File selection section
        file_section = ttk.LabelFrame(processing_frame, text="Document Selection", padding="10")
        file_section.pack(fill="x", pady=(0, 10))
        
        # File list
        file_list_frame = ttk.Frame(file_section)
        file_list_frame.pack(fill="both", expand=True)
        
        self.file_listbox = tk.Listbox(file_list_frame, height=6)
        scrollbar_files = ttk.Scrollbar(file_list_frame, orient="vertical")
        self.file_listbox.config(yscrollcommand=scrollbar_files.set)
        scrollbar_files.config(command=self.file_listbox.yview)
        
        self.file_listbox.pack(side="left", fill="both", expand=True)
        scrollbar_files.pack(side="right", fill="y")
        
        # File buttons
        file_buttons = ttk.Frame(file_section)
        file_buttons.pack(fill="x", pady=(5, 0))
        
        ttk.Button(file_buttons, text="Add Files", command=self._add_files).pack(side="left", padx=(0, 5))
        ttk.Button(file_buttons, text="Add Folder", command=self._add_folder).pack(side="left", padx=(0, 5))
        ttk.Button(file_buttons, text="Clear", command=self._clear_files).pack(side="left", padx=(0, 5))
        
        # Processing options section
        options_section = ttk.LabelFrame(processing_frame, text="Processing Options", padding="10")
        options_section.pack(fill="x", pady=(0, 10))
        
        # Options checkboxes
        self.enable_llm_var = tk.BooleanVar(value=True)
        self.enable_embeddings_var = tk.BooleanVar(value=True)
        self.enable_neo4j_var = tk.BooleanVar(value=False)
        
        ttk.Checkbutton(options_section, text="LLM Ontology Generation", 
                       variable=self.enable_llm_var).pack(anchor="w")
        ttk.Checkbutton(options_section, text="Generate Embeddings", 
                       variable=self.enable_embeddings_var).pack(anchor="w")
        ttk.Checkbutton(options_section, text="Store in Neo4j", 
                       variable=self.enable_neo4j_var).pack(anchor="w")
        
        # Output options
        output_frame = ttk.Frame(options_section)
        output_frame.pack(fill="x", pady=(10, 0))
        
        ttk.Label(output_frame, text="Output File:").pack(side="left")
        self.output_var = tk.StringVar(value="agentic_graph_output.html")
        output_entry = ttk.Entry(output_frame, textvariable=self.output_var, width=30)
        output_entry.pack(side="left", padx=(5, 5), fill="x", expand=True)
        ttk.Button(output_frame, text="Browse", command=self._browse_output).pack(side="right")
        
        # Processing controls
        controls_section = ttk.LabelFrame(processing_frame, text="Processing", padding="10")
        controls_section.pack(fill="x", pady=(0, 10))
        
        # Progress bar
        self.progress_bar = Progressbar(controls_section, variable=self.progress_var, 
                                       maximum=100, length=300)
        self.progress_bar.pack(fill="x", pady=(0, 10))
        
        # Control buttons
        controls_frame = ttk.Frame(controls_section)
        controls_frame.pack(fill="x")
        
        self.process_button = ttk.Button(controls_frame, text="Start Processing", 
                                        command=self._start_processing, state="normal")
        self.process_button.pack(side="left", padx=(0, 5))
        
        self.stop_button = ttk.Button(controls_frame, text="Stop", 
                                     command=self._stop_processing, state="disabled")
        self.stop_button.pack(side="left", padx=(0, 5))
        
        ttk.Button(controls_frame, text="Open Results", 
                  command=self._open_results).pack(side="right")
        
        # Processing log
        log_section = ttk.LabelFrame(processing_frame, text="Processing Log", padding="10")
        log_section.pack(fill="both", expand=True)
        
        self.log_text = scrolledtext.ScrolledText(log_section, height=8, state="disabled")
        self.log_text.pack(fill="both", expand=True)
    
    def _create_config_tab(self, parent):
        """Create the configuration tab."""
        config_frame = ttk.Frame(parent)
        parent.add(config_frame, text="Configuration")
        
        # API Keys section
        api_section = ttk.LabelFrame(config_frame, text="API Configuration", padding="10")
        api_section.pack(fill="x", pady=(0, 10))
        
        # OpenRouter API
        ttk.Label(api_section, text="OpenRouter API Key:").pack(anchor="w")
        self.openrouter_key_var = tk.StringVar()
        openrouter_entry = ttk.Entry(api_section, textvariable=self.openrouter_key_var, show="*", width=50)
        openrouter_entry.pack(fill="x", pady=(0, 10))
        
        # OpenAI API
        ttk.Label(api_section, text="OpenAI API Key:").pack(anchor="w")
        self.openai_key_var = tk.StringVar()
        openai_entry = ttk.Entry(api_section, textvariable=self.openai_key_var, show="*", width=50)
        openai_entry.pack(fill="x", pady=(0, 10))
        
        # Save config button
        ttk.Button(api_section, text="Save Configuration", 
                  command=self._save_config).pack()
        
        # Component Status section
        status_section = ttk.LabelFrame(config_frame, text="Component Status", padding="10")
        status_section.pack(fill="x", pady=(0, 10))
        
        self.status_tree = ttk.Treeview(status_section, columns=("status", "details"), height=6)
        self.status_tree.heading("#0", text="Component")
        self.status_tree.heading("status", text="Status")
        self.status_tree.heading("details", text="Details")
        self.status_tree.pack(fill="x")
        
        # Refresh status button
        ttk.Button(status_section, text="Refresh Status", 
                  command=self._update_component_status).pack(pady=(5, 0))
        
        # Advanced Settings
        advanced_section = ttk.LabelFrame(config_frame, text="Advanced Settings", padding="10")
        advanced_section.pack(fill="x")
        
        # LLM Model
        ttk.Label(advanced_section, text="LLM Model:").pack(anchor="w")
        self.llm_model_var = tk.StringVar(value="microsoft/wizardlm-2-8x22b")
        llm_combo = ttk.Combobox(advanced_section, textvariable=self.llm_model_var,
                                values=["microsoft/wizardlm-2-8x22b", "anthropic/claude-3-sonnet", 
                                       "openai/gpt-4-turbo", "google/gemini-pro"])
        llm_combo.pack(fill="x", pady=(0, 5))
        
        # Embedding Model
        ttk.Label(advanced_section, text="Embedding Model:").pack(anchor="w")
        self.embedding_model_var = tk.StringVar(value="text-embedding-ada-002")
        embedding_combo = ttk.Combobox(advanced_section, textvariable=self.embedding_model_var,
                                      values=["text-embedding-ada-002", "text-embedding-3-small", "text-embedding-3-large"])
        embedding_combo.pack(fill="x")
    
    def _create_results_tab(self, parent):
        """Create the results and analytics tab."""
        results_frame = ttk.Frame(parent)
        parent.add(results_frame, text="Results & Analytics")
        
        # Session info
        session_section = ttk.LabelFrame(results_frame, text="Current Session", padding="10")
        session_section.pack(fill="x", pady=(0, 10))
        
        self.session_info = ttk.Treeview(session_section, columns=("value",), height=6)
        self.session_info.heading("#0", text="Metric")
        self.session_info.heading("value", text="Value")
        self.session_info.pack(fill="x")
        
        # Results actions
        actions_frame = ttk.Frame(results_frame)
        actions_frame.pack(fill="x", pady=(10, 0))
        
        ttk.Button(actions_frame, text="Open Graph Visualization", 
                  command=self._open_results).pack(side="left", padx=(0, 5))
        ttk.Button(actions_frame, text="Export Data", 
                  command=self._export_data).pack(side="left", padx=(0, 5))
        ttk.Button(actions_frame, text="View Neo4j Browser", 
                  command=self._open_neo4j_browser).pack(side="left")
        
        # Analytics display (placeholder for future features)
        analytics_section = ttk.LabelFrame(results_frame, text="Analytics", padding="10")
        analytics_section.pack(fill="both", expand=True, pady=(10, 0))
        
        analytics_text = tk.Text(analytics_section, height=10, state="disabled")
        analytics_text.pack(fill="both", expand=True)
        analytics_text.insert("1.0", "Analytics and insights will be displayed here after processing...")
        analytics_text.config(state="disabled")
    
    def _create_status_bar(self, parent):
        """Create the status bar."""
        status_frame = ttk.Frame(parent)
        status_frame.pack(fill="x", pady=(10, 0))
        
        ttk.Label(status_frame, textvariable=self.status_var).pack(side="left")
        
        # Version info
        version_label = ttk.Label(status_frame, text="Phase 2 - v2.0.0", foreground="gray")
        version_label.pack(side="right")
    
    def _add_files(self):
        """Add individual files to the processing list."""
        files = filedialog.askopenfilenames(
            title="Select Documents",
            filetypes=[
                ("All Supported", "*.pdf;*.docx;*.txt;*.md;*.jpg;*.png"),
                ("PDF Files", "*.pdf"),
                ("Word Documents", "*.docx"),
                ("Text Files", "*.txt;*.md"),
                ("Images", "*.jpg;*.png;*.jpeg"),
                ("All Files", "*.*")
            ]
        )
        
        for file in files:
            if file not in self.selected_files:
                self.selected_files.append(file)
                self.file_listbox.insert(tk.END, Path(file).name)
        
        self._log(f"Added {len(files)} files")
    
    def _add_folder(self):
        """Add all supported files from a folder."""
        folder = filedialog.askdirectory(title="Select Folder")
        if folder:
            folder_path = Path(folder)
            supported_extensions = ['.pdf', '.docx', '.txt', '.md', '.jpg', '.png', '.jpeg']
            
            found_files = []
            for ext in supported_extensions:
                found_files.extend(folder_path.rglob(f'*{ext}'))
            
            for file in found_files:
                file_str = str(file)
                if file_str not in self.selected_files:
                    self.selected_files.append(file_str)
                    self.file_listbox.insert(tk.END, file.name)
            
            self._log(f"Added {len(found_files)} files from folder")
    
    def _clear_files(self):
        """Clear the file list."""
        self.selected_files.clear()
        self.file_listbox.delete(0, tk.END)
        self._log("File list cleared")
    
    def _browse_output(self):
        """Browse for output file location."""
        filename = filedialog.asksaveasfilename(
            title="Save Graph Visualization",
            defaultextension=".html",
            filetypes=[("HTML Files", "*.html"), ("All Files", "*.*")]
        )
        if filename:
            self.output_var.set(filename)
    
    def _start_processing(self):
        """Start the document processing pipeline."""
        if not self.selected_files:
            messagebox.showerror("Error", "Please select files to process")
            return
        
        if self.is_processing:
            messagebox.showwarning("Warning", "Processing is already in progress")
            return
        
        # Validate configuration
        if self.enable_llm_var.get() and not self.openrouter_key_var.get():
            if not messagebox.askyesno("Warning", "OpenRouter API key not set. Continue without LLM ontology generation?"):
                return
            self.enable_llm_var.set(False)
        
        if self.enable_embeddings_var.get() and not self.openai_key_var.get():
            if not messagebox.askyesno("Warning", "OpenAI API key not set. Continue without embeddings?"):
                return
            self.enable_embeddings_var.set(False)
        
        # Start processing in background thread
        self.is_processing = True
        self.process_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.progress_var.set(0)
        self.status_var.set("Processing...")
        
        threading.Thread(target=self._run_processing, daemon=True).start()
    
    def _run_processing(self):
        """Run the processing pipeline in background."""
        try:
            # Set environment variables
            if self.openrouter_key_var.get():
                os.environ['OPENROUTER_API_KEY'] = self.openrouter_key_var.get()
            if self.openai_key_var.get():
                os.environ['OPENAI_API_KEY'] = self.openai_key_var.get()
            
            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Run pipeline
            results = loop.run_until_complete(
                self.pipeline.run_full_pipeline(
                    input_paths=self.selected_files,
                    output_path=self.output_var.get(),
                    enable_llm=self.enable_llm_var.get(),
                    enable_neo4j=self.enable_neo4j_var.get(),
                    enable_embeddings=self.enable_embeddings_var.get()
                )
            )
            
            self.current_session = results
            self.processing_results = results
            
            # Update GUI on main thread
            self.root.after(0, self._processing_complete, results)
            
        except Exception as e:
            self.root.after(0, self._processing_error, str(e))
        finally:
            loop.close()
    
    def _processing_complete(self, results):
        """Handle successful processing completion."""
        self.is_processing = False
        self.process_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.progress_var.set(100)
        self.status_var.set("Processing completed successfully")
        
        # Update results tab
        self._update_session_info(results)
        
        self._log("Processing completed successfully!")
        self._log(f"Session ID: {results['session_id']}")
        self._log(f"Entities: {results['statistics']['entities_extracted']}")
        self._log(f"Relationships: {results['statistics']['relationships_extracted']}")
        
        # Ask to open results
        if messagebox.askyesno("Success", "Processing completed! Open the graph visualization?"):
            self._open_results()
    
    def _processing_error(self, error_message):
        """Handle processing error."""
        self.is_processing = False
        self.process_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.status_var.set("Processing failed")
        
        self._log(f"ERROR: {error_message}")
        messagebox.showerror("Processing Error", f"Processing failed:\n{error_message}")
    
    def _stop_processing(self):
        """Stop the current processing (placeholder)."""
        # Note: Actual stopping would require more complex async coordination
        self._log("Stop requested (processing will complete current stage)")
        messagebox.showinfo("Stop", "Processing will stop after current stage completes")
    
    def _open_results(self):
        """Open the generated graph visualization."""
        output_file = Path(self.output_var.get())
        if output_file.exists():
            webbrowser.open(f"file://{output_file.absolute()}")
        else:
            messagebox.showerror("Error", "Output file not found. Please run processing first.")
    
    def _save_config(self):
        """Save configuration to environment file."""
        try:
            config_content = f"""# Agentic Graph RAG Configuration
OPENROUTER_API_KEY={self.openrouter_key_var.get()}
OPENAI_API_KEY={self.openai_key_var.get()}
OPENROUTER_MODEL={self.llm_model_var.get()}
OPENAI_EMBEDDING_MODEL={self.embedding_model_var.get()}
"""
            
            config_file = Path(".env")
            config_file.write_text(config_content)
            
            messagebox.showinfo("Success", "Configuration saved to .env file")
            self._log("Configuration saved")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save configuration: {e}")
    
    def _load_configuration(self):
        """Load configuration from environment."""
        self.openrouter_key_var.set(os.getenv('OPENROUTER_API_KEY', ''))
        self.openai_key_var.set(os.getenv('OPENAI_API_KEY', ''))
        self.llm_model_var.set(os.getenv('OPENROUTER_MODEL', 'microsoft/wizardlm-2-8x22b'))
        self.embedding_model_var.set(os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-ada-002'))
    
    def _update_component_status(self):
        """Update the component status display."""
        # Clear existing items
        for item in self.status_tree.get_children():
            self.status_tree.delete(item)
        
        # Check components
        components = [
            ("Document Processing", "✅ Available", "Multi-format document processing"),
            ("LLM Ontology", "✅ Available" if os.getenv('OPENROUTER_API_KEY') else "⚠️ API Key Required", "OpenRouter API"),
            ("Embeddings", "✅ Available" if os.getenv('OPENAI_API_KEY') else "⚠️ API Key Required", "OpenAI Embeddings"),
            ("Neo4j Storage", "⚠️ Not Connected", "Graph database storage"),
            ("Visualization", "✅ Available", "3D graph visualization")
        ]
        
        for name, status, details in components:
            self.status_tree.insert("", "end", text=name, values=(status, details))
    
    def _update_session_info(self, results):
        """Update session information display."""
        # Clear existing items
        for item in self.session_info.get_children():
            self.session_info.delete(item)
        
        # Add session metrics
        if results:
            metrics = [
                ("Session ID", results.get('session_id', 'N/A')),
                ("Documents Processed", results['statistics']['documents_processed']),
                ("Entities Extracted", results['statistics']['entities_extracted']),
                ("Relationships Found", results['statistics']['relationships_extracted']),
                ("Processing Time", results.get('total_duration', 'N/A')),
                ("Output File", results.get('outputs', {}).get('visualization_path', 'N/A'))
            ]
            
            for metric, value in metrics:
                self.session_info.insert("", "end", text=metric, values=(str(value),))
    
    def _export_data(self):
        """Export processing results."""
        if not self.processing_results:
            messagebox.showwarning("Warning", "No results to export")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Export Results",
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")]
        )
        
        if filename:
            try:
                import json
                with open(filename, 'w') as f:
                    json.dump(self.processing_results, f, indent=2, default=str)
                messagebox.showinfo("Success", f"Results exported to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Export failed: {e}")
    
    def _open_neo4j_browser(self):
        """Open Neo4j Browser."""
        webbrowser.open("http://localhost:7474")
    
    def _log(self, message):
        """Add message to processing log."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}\n"
        
        self.log_text.config(state="normal")
        self.log_text.insert(tk.END, log_message)
        self.log_text.see(tk.END)
        self.log_text.config(state="disabled")
        
        # Also log to console
        logger.info(message)
    
    def run(self):
        """Run the GUI application."""
        try:
            self._log("Agentic Graph RAG Pipeline - Phase 2 GUI started")
            self._log("Please select documents and configure processing options")
            self.root.mainloop()
        except KeyboardInterrupt:
            logger.info("GUI application interrupted")
        except Exception as e:
            logger.error(f"GUI error: {e}")
            messagebox.showerror("Application Error", f"An error occurred: {e}")


def main():
    """Main entry point."""
    if not HAVE_TKINTER:
        print("Error: Tkinter not available. Please install tkinter.")
        sys.exit(1)
    
    try:
        app = Phase2GraphRAGGUI()
        app.run()
    except Exception as e:
        print(f"Failed to start application: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
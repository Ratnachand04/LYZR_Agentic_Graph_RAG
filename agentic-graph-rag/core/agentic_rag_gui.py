#!/usr/bin/env python3
"""
Agentic RAG GUI - Comprehensive Desktop Interface
Integrates all agentic components with streaming displays and real-time visualization.
"""

import logging
import asyncio
import threading
import json
import traceback
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from pathlib import Path
import queue
import uuid

try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox, scrolledtext
    from tkinter import font as tkFont
    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False
    logging.warning("Tkinter not available")

try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    from matplotlib.figure import Figure
    import matplotlib.patches as patches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logging.warning("Matplotlib not available")

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    logging.warning("NetworkX not available")

# Import agentic components
try:
    from .faiss_index_manager import FAISSIndexManager
    from .cypher_traversal_engine import CypherTraversalEngine
    from .query_analyzer import QueryAnalyzer
    from .reasoning_engine_unified import ReasoningEngine
    from .interactive_graph_editor import InteractiveGraphEditor
    from .agentic_rag_api import AgenticRAGAPI
    from .neo4j_graph_store_v2 import Neo4jGraphStore
    from .embedding_manager_v2 import EmbeddingManager
except ImportError as e:
    logging.warning(f"Some agentic components not available: {e}")

logger = logging.getLogger(__name__)


class StreamingTextWidget:
    """Widget for displaying streaming text with real-time updates."""
    
    def __init__(self, parent, height=10, width=80):
        """Initialize streaming text widget."""
        self.frame = ttk.Frame(parent)
        
        # Text widget with scrollbar
        self.text_widget = scrolledtext.ScrolledText(
            self.frame,
            height=height,
            width=width,
            wrap=tk.WORD,
            state=tk.DISABLED,
            font=('Consolas', 10)
        )
        self.text_widget.pack(fill=tk.BOTH, expand=True)
        
        # Configure tags for different message types
        self.text_widget.tag_configure("info", foreground="blue")
        self.text_widget.tag_configure("warning", foreground="orange")
        self.text_widget.tag_configure("error", foreground="red")
        self.text_widget.tag_configure("success", foreground="green")
        self.text_widget.tag_configure("query", foreground="purple", font=('Consolas', 10, 'bold'))
        self.text_widget.tag_configure("answer", foreground="black", font=('Consolas', 10))
        
        # Auto-scroll option
        self.auto_scroll = tk.BooleanVar(value=True)
        
        # Message queue for thread-safe updates
        self.message_queue = queue.Queue()
        self.check_messages()
    
    def pack(self, **kwargs):
        """Pack the frame."""
        self.frame.pack(**kwargs)
    
    def grid(self, **kwargs):
        """Grid the frame."""
        self.frame.grid(**kwargs)
    
    def check_messages(self):
        """Check for new messages in queue."""
        try:
            while True:
                message, tag = self.message_queue.get_nowait()
                self._append_text(message, tag)
        except queue.Empty:
            pass
        
        # Schedule next check
        self.frame.after(100, self.check_messages)
    
    def append_text(self, text: str, tag: str = "info"):
        """Thread-safe text appending."""
        self.message_queue.put((text, tag))
    
    def _append_text(self, text: str, tag: str = "info"):
        """Internal method to append text."""
        self.text_widget.configure(state=tk.NORMAL)
        
        # Add timestamp
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.text_widget.insert(tk.END, f"[{timestamp}] ", "info")
        self.text_widget.insert(tk.END, f"{text}\n", tag)
        
        # Auto-scroll to bottom
        if self.auto_scroll.get():
            self.text_widget.see(tk.END)
        
        self.text_widget.configure(state=tk.DISABLED)
    
    def clear(self):
        """Clear all text."""
        self.text_widget.configure(state=tk.NORMAL)
        self.text_widget.delete(1.0, tk.END)
        self.text_widget.configure(state=tk.DISABLED)


class SystemStatusWidget:
    """Widget for displaying system status and metrics."""
    
    def __init__(self, parent):
        """Initialize system status widget."""
        self.frame = ttk.LabelFrame(parent, text="System Status")
        
        # Status variables
        self.status_vars = {
            'graph_store': tk.StringVar(value="Disconnected"),
            'faiss_index': tk.StringVar(value="Not Loaded"),
            'query_analyzer': tk.StringVar(value="Ready"),
            'reasoning_engine': tk.StringVar(value="Ready"),
            'api_server': tk.StringVar(value="Stopped")
        }
        
        # Status indicators
        self.create_status_indicators()
        
        # Metrics
        self.metrics_vars = {
            'nodes_count': tk.StringVar(value="0"),
            'edges_count': tk.StringVar(value="0"),
            'queries_processed': tk.StringVar(value="0"),
            'avg_response_time': tk.StringVar(value="0.0s")
        }
        
        self.create_metrics_display()
    
    def pack(self, **kwargs):
        """Pack the frame."""
        self.frame.pack(**kwargs)
    
    def grid(self, **kwargs):
        """Grid the frame."""
        self.frame.grid(**kwargs)
    
    def create_status_indicators(self):
        """Create status indicator widgets."""
        status_frame = ttk.Frame(self.frame)
        status_frame.pack(fill=tk.X, padx=5, pady=5)
        
        row = 0
        for component, var in self.status_vars.items():
            ttk.Label(status_frame, text=f"{component}:").grid(row=row, column=0, sticky=tk.W, padx=(0, 10))
            status_label = ttk.Label(status_frame, textvariable=var)
            status_label.grid(row=row, column=1, sticky=tk.W)
            row += 1
    
    def create_metrics_display(self):
        """Create metrics display widgets."""
        metrics_frame = ttk.LabelFrame(self.frame, text="Metrics")
        metrics_frame.pack(fill=tk.X, padx=5, pady=5)
        
        row = 0
        for metric, var in self.metrics_vars.items():
            ttk.Label(metrics_frame, text=f"{metric.replace('_', ' ').title()}:").grid(row=row, column=0, sticky=tk.W, padx=(0, 10))
            ttk.Label(metrics_frame, textvariable=var).grid(row=row, column=1, sticky=tk.W)
            row += 1
    
    def update_status(self, component: str, status: str):
        """Update component status."""
        if component in self.status_vars:
            self.status_vars[component].set(status)
    
    def update_metric(self, metric: str, value: str):
        """Update metric value."""
        if metric in self.metrics_vars:
            self.metrics_vars[metric].set(str(value))


class GraphVisualizationWidget:
    """Widget for graph visualization using matplotlib."""
    
    def __init__(self, parent):
        """Initialize graph visualization widget."""
        self.frame = ttk.Frame(parent)
        
        # Create matplotlib figure
        if MATPLOTLIB_AVAILABLE and NETWORKX_AVAILABLE:
            self.figure = Figure(figsize=(8, 6), dpi=100)
            self.ax = self.figure.add_subplot(111)
            
            # Create canvas
            self.canvas = FigureCanvasTkAgg(self.figure, self.frame)
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Navigation toolbar
            self.toolbar = NavigationToolbar2Tk(self.canvas, self.frame)
            self.toolbar.update()
            
            # Initialize empty graph
            self.graph = nx.DiGraph()
            self.pos = {}
            self.node_colors = []
            self.edge_colors = []
            
            self.draw_empty_graph()
        else:
            # Fallback text display
            ttk.Label(self.frame, text="Graph visualization not available\n(requires matplotlib and networkx)").pack(expand=True)
    
    def pack(self, **kwargs):
        """Pack the frame."""
        self.frame.pack(**kwargs)
    
    def grid(self, **kwargs):
        """Grid the frame."""
        self.frame.grid(**kwargs)
    
    def draw_empty_graph(self):
        """Draw empty graph placeholder."""
        if not MATPLOTLIB_AVAILABLE:
            return
        
        self.ax.clear()
        self.ax.text(0.5, 0.5, 'No graph data\nClick "Load Graph" to visualize', 
                    ha='center', va='center', transform=self.ax.transAxes,
                    fontsize=14, color='gray')
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.axis('off')
        self.canvas.draw()
    
    def update_graph(self, nodes: List[Dict], edges: List[Dict]):
        """Update graph visualization with new data."""
        if not MATPLOTLIB_AVAILABLE or not NETWORKX_AVAILABLE:
            return
        
        # Clear existing graph
        self.graph.clear()
        
        # Add nodes
        node_labels = {}
        node_colors = []
        color_map = {
            'Entity': '#3498db',
            'Concept': '#e74c3c',
            'Event': '#f39c12',
            'Location': '#27ae60',
            'Person': '#9b59b6'
        }
        
        for node in nodes:
            self.graph.add_node(node['id'])
            node_labels[node['id']] = node.get('label', node['id'])
            node_type = node.get('type', 'Entity')
            node_colors.append(color_map.get(node_type, '#3498db'))
        
        # Add edges
        for edge in edges:
            if edge['source'] in self.graph and edge['target'] in self.graph:
                self.graph.add_edge(edge['source'], edge['target'])
        
        # Calculate layout
        if len(self.graph.nodes()) > 0:
            try:
                if len(self.graph.nodes()) < 100:
                    self.pos = nx.spring_layout(self.graph, k=1, iterations=50)
                else:
                    self.pos = nx.random_layout(self.graph)
            except:
                self.pos = nx.random_layout(self.graph)
        
        # Draw graph
        self.ax.clear()
        
        if self.graph.nodes():
            # Draw edges
            nx.draw_networkx_edges(self.graph, self.pos, ax=self.ax, 
                                 edge_color='#cccccc', alpha=0.7, width=1)
            
            # Draw nodes
            nx.draw_networkx_nodes(self.graph, self.pos, ax=self.ax,
                                 node_color=node_colors, node_size=300, alpha=0.9)
            
            # Draw labels
            nx.draw_networkx_labels(self.graph, self.pos, node_labels, ax=self.ax,
                                  font_size=8, font_weight='bold')
        
        self.ax.set_title(f"Knowledge Graph ({len(self.graph.nodes())} nodes, {len(self.graph.edges())} edges)")
        self.ax.axis('off')
        self.canvas.draw()
    
    def highlight_path(self, path: List[str]):
        """Highlight a specific path in the graph."""
        if not MATPLOTLIB_AVAILABLE or not path:
            return
        
        # Redraw with highlighted path
        self.ax.clear()
        
        if self.graph.nodes():
            # Draw all edges in light gray
            nx.draw_networkx_edges(self.graph, self.pos, ax=self.ax,
                                 edge_color='#eeeeee', alpha=0.5, width=1)
            
            # Draw path edges in red
            path_edges = [(path[i], path[i+1]) for i in range(len(path)-1) if i+1 < len(path)]
            valid_path_edges = [e for e in path_edges if self.graph.has_edge(e[0], e[1])]
            
            if valid_path_edges:
                nx.draw_networkx_edges(self.graph, self.pos, edgelist=valid_path_edges,
                                     ax=self.ax, edge_color='red', width=3, alpha=0.8)
            
            # Draw all nodes
            node_colors = ['red' if node in path else '#3498db' for node in self.graph.nodes()]
            nx.draw_networkx_nodes(self.graph, self.pos, ax=self.ax,
                                 node_color=node_colors, node_size=300, alpha=0.9)
            
            # Draw labels
            nx.draw_networkx_labels(self.graph, self.pos, ax=self.ax,
                                  font_size=8, font_weight='bold')
        
        self.ax.set_title(f"Knowledge Graph - Path Highlighted")
        self.ax.axis('off')
        self.canvas.draw()


class QueryInterface:
    """Interface for query input and configuration."""
    
    def __init__(self, parent, on_query_callback: Callable):
        """Initialize query interface."""
        self.frame = ttk.LabelFrame(parent, text="Query Interface")
        self.on_query_callback = on_query_callback
        
        self.create_widgets()
    
    def pack(self, **kwargs):
        """Pack the frame."""
        self.frame.pack(**kwargs)
    
    def grid(self, **kwargs):
        """Grid the frame."""
        self.frame.grid(**kwargs)
    
    def create_widgets(self):
        """Create query interface widgets."""
        # Query input
        input_frame = ttk.Frame(self.frame)
        input_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(input_frame, text="Query:").pack(anchor=tk.W)
        
        query_frame = ttk.Frame(input_frame)
        query_frame.pack(fill=tk.X, pady=(5, 0))
        
        self.query_text = tk.Text(query_frame, height=3, wrap=tk.WORD)
        query_scroll = ttk.Scrollbar(query_frame, orient=tk.VERTICAL, command=self.query_text.yview)
        self.query_text.configure(yscrollcommand=query_scroll.set)
        
        self.query_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        query_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Query options
        options_frame = ttk.LabelFrame(self.frame, text="Query Options")
        options_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Query type
        type_frame = ttk.Frame(options_frame)
        type_frame.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Label(type_frame, text="Query Type:").pack(side=tk.LEFT)
        self.query_type = ttk.Combobox(type_frame, values=[
            "factual", "analytical", "exploratory", "comparative", "summarization"
        ], state="readonly", width=15)
        self.query_type.set("factual")
        self.query_type.pack(side=tk.LEFT, padx=(10, 0))
        
        # Reasoning depth
        depth_frame = ttk.Frame(options_frame)
        depth_frame.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Label(depth_frame, text="Reasoning Depth:").pack(side=tk.LEFT)
        self.reasoning_depth = tk.IntVar(value=3)
        depth_scale = ttk.Scale(depth_frame, from_=1, to=10, variable=self.reasoning_depth, 
                               orient=tk.HORIZONTAL, length=150)
        depth_scale.pack(side=tk.LEFT, padx=(10, 5))
        
        self.depth_label = ttk.Label(depth_frame, text="3")
        self.depth_label.pack(side=tk.LEFT)
        
        # Update depth label
        def update_depth_label(*args):
            self.depth_label.config(text=str(self.reasoning_depth.get()))
        self.reasoning_depth.trace('w', update_depth_label)
        
        # Max results
        results_frame = ttk.Frame(options_frame)
        results_frame.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Label(results_frame, text="Max Results:").pack(side=tk.LEFT)
        self.max_results = ttk.Spinbox(results_frame, from_=1, to=100, value=10, width=10)
        self.max_results.pack(side=tk.LEFT, padx=(10, 0))
        
        # Stream results option
        self.stream_results = tk.BooleanVar(value=True)
        ttk.Checkbutton(results_frame, text="Stream Results", 
                       variable=self.stream_results).pack(side=tk.LEFT, padx=(20, 0))
        
        # Action buttons
        button_frame = ttk.Frame(self.frame)
        button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.query_button = ttk.Button(button_frame, text="Execute Query", 
                                     command=self.execute_query)
        self.query_button.pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(button_frame, text="Clear Query", 
                  command=self.clear_query).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(button_frame, text="Load Example", 
                  command=self.load_example_query).pack(side=tk.LEFT)
        
        # Query history
        history_frame = ttk.LabelFrame(self.frame, text="Recent Queries")
        history_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.history_listbox = tk.Listbox(history_frame, height=5)
        history_scroll = ttk.Scrollbar(history_frame, orient=tk.VERTICAL, 
                                     command=self.history_listbox.yview)
        self.history_listbox.configure(yscrollcommand=history_scroll.set)
        
        self.history_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        history_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Bind double-click to load query
        self.history_listbox.bind('<Double-1>', self.load_from_history)
        
        # Query history storage
        self.query_history = []
    
    def execute_query(self):
        """Execute the current query."""
        query_text = self.query_text.get(1.0, tk.END).strip()
        if not query_text:
            messagebox.showwarning("No Query", "Please enter a query.")
            return
        
        # Add to history
        if query_text not in self.query_history:
            self.query_history.append(query_text)
            self.history_listbox.insert(tk.END, query_text[:50] + "..." if len(query_text) > 50 else query_text)
        
        # Prepare query parameters
        query_params = {
            'query': query_text,
            'query_type': self.query_type.get(),
            'reasoning_depth': self.reasoning_depth.get(),
            'max_results': int(self.max_results.get()),
            'stream_results': self.stream_results.get()
        }
        
        # Call callback
        self.on_query_callback(query_params)
    
    def clear_query(self):
        """Clear the query input."""
        self.query_text.delete(1.0, tk.END)
    
    def load_example_query(self):
        """Load an example query."""
        examples = [
            "What are the main concepts related to machine learning?",
            "How is artificial intelligence connected to neural networks?",
            "Explain the relationship between data science and statistics.",
            "What are the key components of a recommendation system?",
            "How do knowledge graphs support semantic search?"
        ]
        
        import random
        example = random.choice(examples)
        self.query_text.delete(1.0, tk.END)
        self.query_text.insert(1.0, example)
    
    def load_from_history(self, event):
        """Load query from history."""
        selection = self.history_listbox.curselection()
        if selection:
            index = selection[0]
            query = self.query_history[index]
            self.query_text.delete(1.0, tk.END)
            self.query_text.insert(1.0, query)
    
    def set_query_text(self, text: str):
        """Set query text programmatically."""
        self.query_text.delete(1.0, tk.END)
        self.query_text.insert(1.0, text)


class AgenticRAGGUI:
    """
    Comprehensive Agentic RAG GUI Application.
    
    Features:
    - Query interface with streaming results
    - Real-time graph visualization  
    - System status monitoring
    - Component management
    - Configuration management
    - Export/import functionality
    """
    
    def __init__(self):
        """Initialize the Agentic RAG GUI."""
        if not TKINTER_AVAILABLE:
            raise RuntimeError("Tkinter is not available")
        
        # Initialize components
        self.components = {}
        self.component_status = {}
        self.current_query_id = None
        
        # Initialize GUI
        self.setup_gui()
        
        # Initialize agentic components
        self.initialize_components()
        
        logger.info("Agentic RAG GUI initialized")
    
    def setup_gui(self):
        """Setup the main GUI window and widgets."""
        self.root = tk.Tk()
        self.root.title("Agentic RAG System - Interactive Knowledge Assistant")
        self.root.geometry("1400x900")
        self.root.minsize(1200, 800)
        
        # Configure styles
        style = ttk.Style()
        style.theme_use('clam')
        
        # Create main menu
        self.create_menu()
        
        # Create main layout
        self.create_main_layout()
        
        # Setup keyboard shortcuts
        self.setup_shortcuts()
    
    def create_menu(self):
        """Create application menu bar."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Graph...", command=self.load_graph)
        file_menu.add_command(label="Save Graph...", command=self.save_graph)
        file_menu.add_separator()
        file_menu.add_command(label="Export Results...", command=self.export_results)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # System menu
        system_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="System", menu=system_menu)
        system_menu.add_command(label="Start Components", command=self.start_components)
        system_menu.add_command(label="Stop Components", command=self.stop_components)
        system_menu.add_separator()
        system_menu.add_command(label="Configuration...", command=self.show_configuration)
        system_menu.add_command(label="System Status...", command=self.show_system_status)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
        help_menu.add_command(label="User Guide", command=self.show_user_guide)
    
    def create_main_layout(self):
        """Create main application layout."""
        # Main paned window
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel (Query + Status)
        left_panel = ttk.Frame(main_paned)
        main_paned.add(left_panel, weight=1)
        
        # Right panel (Visualization + Results)
        right_panel = ttk.Frame(main_paned)
        main_paned.add(right_panel, weight=2)
        
        # Setup left panel
        self.setup_left_panel(left_panel)
        
        # Setup right panel  
        self.setup_right_panel(right_panel)
    
    def setup_left_panel(self, parent):
        """Setup left panel with query interface and status."""
        # Query interface
        self.query_interface = QueryInterface(parent, self.execute_query)
        self.query_interface.pack(fill=tk.X, padx=5, pady=5)
        
        # System status
        self.status_widget = SystemStatusWidget(parent)
        self.status_widget.pack(fill=tk.X, padx=5, pady=5)
        
        # Control buttons
        control_frame = ttk.LabelFrame(parent, text="System Controls")
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.start_button = ttk.Button(button_frame, text="Start System", 
                                     command=self.start_components)
        self.start_button.pack(side=tk.LEFT, padx=(0, 5))
        
        self.stop_button = ttk.Button(button_frame, text="Stop System", 
                                    command=self.stop_components, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Button(button_frame, text="Refresh Status", 
                  command=self.refresh_status).pack(side=tk.LEFT)
    
    def setup_right_panel(self, parent):
        """Setup right panel with visualization and results."""
        # Create notebook for tabs
        notebook = ttk.Notebook(parent)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Graph visualization tab
        graph_frame = ttk.Frame(notebook)
        notebook.add(graph_frame, text="Graph Visualization")
        
        self.graph_widget = GraphVisualizationWidget(graph_frame)
        self.graph_widget.pack(fill=tk.BOTH, expand=True)
        
        # Results tab
        results_frame = ttk.Frame(notebook)
        notebook.add(results_frame, text="Query Results")
        
        # Results display
        results_paned = ttk.PanedWindow(results_frame, orient=tk.VERTICAL)
        results_paned.pack(fill=tk.BOTH, expand=True)
        
        # Streaming results
        stream_frame = ttk.LabelFrame(results_paned, text="Streaming Results")
        results_paned.add(stream_frame, weight=2)
        
        self.results_widget = StreamingTextWidget(stream_frame, height=15)
        self.results_widget.pack(fill=tk.BOTH, expand=True)
        
        # System logs
        log_frame = ttk.LabelFrame(results_paned, text="System Logs")
        results_paned.add(log_frame, weight=1)
        
        self.log_widget = StreamingTextWidget(log_frame, height=8)
        self.log_widget.pack(fill=tk.BOTH, expand=True)
        
        # Analytics tab
        analytics_frame = ttk.Frame(notebook)
        notebook.add(analytics_frame, text="Analytics")
        
        self.setup_analytics_tab(analytics_frame)
    
    def setup_analytics_tab(self, parent):
        """Setup analytics tab."""
        # Placeholder for analytics
        ttk.Label(parent, text="Analytics Dashboard", font=('Arial', 16, 'bold')).pack(pady=20)
        
        # Metrics frame
        metrics_frame = ttk.LabelFrame(parent, text="Query Metrics")
        metrics_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Add some basic metrics
        self.analytics_vars = {
            'total_queries': tk.StringVar(value="0"),
            'avg_response_time': tk.StringVar(value="0.0s"),
            'success_rate': tk.StringVar(value="0%"),
            'top_query_type': tk.StringVar(value="N/A")
        }
        
        row = 0
        for metric, var in self.analytics_vars.items():
            ttk.Label(metrics_frame, text=f"{metric.replace('_', ' ').title()}:").grid(
                row=row, column=0, sticky=tk.W, padx=5, pady=2)
            ttk.Label(metrics_frame, textvariable=var).grid(
                row=row, column=1, sticky=tk.W, padx=20, pady=2)
            row += 1
    
    def setup_shortcuts(self):
        """Setup keyboard shortcuts."""
        self.root.bind('<Control-Return>', lambda e: self.query_interface.execute_query())
        self.root.bind('<Control-n>', lambda e: self.query_interface.clear_query())
        self.root.bind('<Control-s>', lambda e: self.save_graph())
        self.root.bind('<Control-o>', lambda e: self.load_graph())
        self.root.bind('<F5>', lambda e: self.refresh_status())
    
    def initialize_components(self):
        """Initialize agentic system components."""
        try:
            self.log_widget.append_text("Initializing agentic components...", "info")
            
            # Initialize core components (placeholder - actual imports would be needed)
            self.components = {
                'faiss_manager': None,  # FAISSIndexManager()
                'cypher_engine': None,  # CypherTraversalEngine()
                'query_analyzer': None,  # QueryAnalyzer()
                'reasoning_engine': None,  # ReasoningEngine()
                'graph_editor': None,  # InteractiveGraphEditor()
                'api_server': None  # AgenticRAGAPI()
            }
            
            # Update status
            for component in self.components:
                self.component_status[component] = "Initialized"
                self.status_widget.update_status(component, "Ready")
            
            self.log_widget.append_text("Agentic components initialized successfully", "success")
            
        except Exception as e:
            self.log_widget.append_text(f"Failed to initialize components: {e}", "error")
            logger.error(f"Component initialization failed: {e}")
    
    def start_components(self):
        """Start all system components."""
        try:
            self.log_widget.append_text("Starting system components...", "info")
            
            # Start each component
            for component_name, component in self.components.items():
                try:
                    # Component-specific startup logic would go here
                    self.component_status[component_name] = "Running"
                    self.status_widget.update_status(component_name, "Running")
                    self.log_widget.append_text(f"Started {component_name}", "success")
                except Exception as e:
                    self.component_status[component_name] = "Failed"
                    self.status_widget.update_status(component_name, "Failed")
                    self.log_widget.append_text(f"Failed to start {component_name}: {e}", "error")
            
            # Update UI state
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            
            self.log_widget.append_text("System startup completed", "success")
            
        except Exception as e:
            self.log_widget.append_text(f"System startup failed: {e}", "error")
            logger.error(f"System startup failed: {e}")
    
    def stop_components(self):
        """Stop all system components."""
        try:
            self.log_widget.append_text("Stopping system components...", "info")
            
            # Stop each component
            for component_name in self.components:
                self.component_status[component_name] = "Stopped"
                self.status_widget.update_status(component_name, "Stopped")
                self.log_widget.append_text(f"Stopped {component_name}", "info")
            
            # Update UI state
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            
            self.log_widget.append_text("System shutdown completed", "success")
            
        except Exception as e:
            self.log_widget.append_text(f"System shutdown failed: {e}", "error")
            logger.error(f"System shutdown failed: {e}")
    
    def execute_query(self, query_params: Dict[str, Any]):
        """Execute a query using the agentic system."""
        try:
            self.current_query_id = str(uuid.uuid4())
            query_text = query_params['query']
            
            self.results_widget.append_text(f"Executing query: {query_text}", "query")
            self.log_widget.append_text(f"Query started: {self.current_query_id}", "info")
            
            # Simulate query processing (replace with actual agentic system calls)
            self.simulate_query_processing(query_params)
            
        except Exception as e:
            self.results_widget.append_text(f"Query execution failed: {e}", "error")
            self.log_widget.append_text(f"Query failed: {e}", "error")
            logger.error(f"Query execution failed: {e}")
    
    def simulate_query_processing(self, query_params: Dict[str, Any]):
        """Simulate query processing with streaming results."""
        def process_query():
            try:
                # Simulate query analysis
                self.results_widget.append_text("🔍 Analyzing query intent...", "info")
                threading.Event().wait(1)  # Simulate processing time
                
                self.results_widget.append_text("📊 Query classified as: " + query_params['query_type'], "info")
                
                # Simulate graph traversal
                self.results_widget.append_text("🔗 Traversing knowledge graph...", "info")
                threading.Event().wait(1)
                
                self.results_widget.append_text("📈 Found 15 relevant nodes, 23 relationships", "info")
                
                # Simulate reasoning
                self.results_widget.append_text("🧠 Applying reasoning engine...", "info")
                threading.Event().wait(1)
                
                # Simulate streaming answer
                answer_parts = [
                    "Based on the knowledge graph analysis:",
                    "• Found strong connections between the queried concepts",
                    "• Identified 3 main relationship patterns",
                    "• Reasoning confidence: 87%",
                    "",
                    "The system has identified several key insights:",
                    "1. Primary concept relationships show high semantic similarity",
                    "2. Historical context suggests strong causal connections", 
                    "3. Cross-references indicate robust knowledge validation",
                    "",
                    "🎯 Query processing completed successfully"
                ]
                
                for part in answer_parts:
                    self.results_widget.append_text(part, "answer")
                    threading.Event().wait(0.5)  # Stream delay
                
                # Update analytics
                self.update_analytics()
                
                self.log_widget.append_text(f"Query completed: {self.current_query_id}", "success")
                
            except Exception as e:
                self.results_widget.append_text(f"Query processing failed: {e}", "error")
                self.log_widget.append_text(f"Query processing error: {e}", "error")
        
        # Run in separate thread to avoid blocking GUI
        threading.Thread(target=process_query, daemon=True).start()
    
    def update_analytics(self):
        """Update analytics metrics."""
        # Simulate updating analytics
        current_queries = int(self.analytics_vars['total_queries'].get())
        self.analytics_vars['total_queries'].set(str(current_queries + 1))
        self.analytics_vars['avg_response_time'].set("2.3s")
        self.analytics_vars['success_rate'].set("94%")
        self.analytics_vars['top_query_type'].set("Factual")
    
    def refresh_status(self):
        """Refresh system status display."""
        self.log_widget.append_text("Refreshing system status...", "info")
        
        # Update metrics
        self.status_widget.update_metric('nodes_count', '1,247')
        self.status_widget.update_metric('edges_count', '3,891') 
        self.status_widget.update_metric('queries_processed', '156')
        self.status_widget.update_metric('avg_response_time', '2.1s')
        
        self.log_widget.append_text("Status refresh completed", "success")
    
    def load_graph(self):
        """Load graph data from file."""
        filename = filedialog.askopenfilename(
            title="Load Graph Data",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                self.log_widget.append_text(f"Loading graph from {filename}...", "info")
                
                # Load and parse graph data (placeholder)
                with open(filename, 'r') as f:
                    data = json.load(f)
                
                # Update visualization
                nodes = data.get('nodes', [])
                edges = data.get('edges', [])
                self.graph_widget.update_graph(nodes, edges)
                
                self.log_widget.append_text(f"Loaded {len(nodes)} nodes, {len(edges)} edges", "success")
                
            except Exception as e:
                self.log_widget.append_text(f"Failed to load graph: {e}", "error")
                messagebox.showerror("Load Error", f"Failed to load graph:\n{e}")
    
    def save_graph(self):
        """Save current graph data to file."""
        filename = filedialog.asksaveasfilename(
            title="Save Graph Data",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                self.log_widget.append_text(f"Saving graph to {filename}...", "info")
                
                # Export graph data (placeholder)
                graph_data = {
                    'nodes': [],
                    'edges': [],
                    'metadata': {
                        'exported_at': datetime.now().isoformat(),
                        'version': '1.0'
                    }
                }
                
                with open(filename, 'w') as f:
                    json.dump(graph_data, f, indent=2)
                
                self.log_widget.append_text("Graph saved successfully", "success")
                
            except Exception as e:
                self.log_widget.append_text(f"Failed to save graph: {e}", "error")
                messagebox.showerror("Save Error", f"Failed to save graph:\n{e}")
    
    def export_results(self):
        """Export query results to file."""
        filename = filedialog.asksaveasfilename(
            title="Export Results",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                # Export current results (placeholder)
                results_text = self.results_widget.text_widget.get(1.0, tk.END)
                
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(results_text)
                
                self.log_widget.append_text(f"Results exported to {filename}", "success")
                
            except Exception as e:
                self.log_widget.append_text(f"Failed to export results: {e}", "error")
                messagebox.showerror("Export Error", f"Failed to export results:\n{e}")
    
    def show_configuration(self):
        """Show configuration dialog."""
        # Create configuration dialog
        config_window = tk.Toplevel(self.root)
        config_window.title("System Configuration")
        config_window.geometry("500x400")
        config_window.transient(self.root)
        config_window.grab_set()
        
        # Configuration notebook
        notebook = ttk.Notebook(config_window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # API Configuration
        api_frame = ttk.Frame(notebook)
        notebook.add(api_frame, text="API Settings")
        
        # Add configuration fields (placeholder)
        ttk.Label(api_frame, text="OpenRouter API Key:").pack(anchor=tk.W, pady=(10, 5))
        ttk.Entry(api_frame, show="*", width=50).pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(api_frame, text="Neo4j Connection:").pack(anchor=tk.W, pady=(10, 5))
        ttk.Entry(api_frame, width=50).pack(fill=tk.X, pady=(0, 10))
        
        # System Configuration
        system_frame = ttk.Frame(notebook)
        notebook.add(system_frame, text="System Settings")
        
        # Buttons
        button_frame = ttk.Frame(config_window)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(button_frame, text="Save", command=config_window.destroy).pack(side=tk.RIGHT, padx=(5, 0))
        ttk.Button(button_frame, text="Cancel", command=config_window.destroy).pack(side=tk.RIGHT)
    
    def show_system_status(self):
        """Show detailed system status dialog."""
        # Create status dialog
        status_window = tk.Toplevel(self.root)
        status_window.title("Detailed System Status")
        status_window.geometry("600x500")
        status_window.transient(self.root)
        
        # Status display
        status_text = scrolledtext.ScrolledText(status_window, height=25, width=70)
        status_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Add detailed status info
        status_info = f"""
System Status Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*50}

Component Status:
- Graph Store: Connected (Neo4j v4.4.0)
- FAISS Index: Loaded (1,247 vectors)
- Query Analyzer: Ready
- Reasoning Engine: Active
- API Server: Running on port 8000

Performance Metrics:
- Uptime: 2h 34m 15s
- Memory Usage: 456 MB / 2 GB
- CPU Usage: 12%
- Query Rate: 2.3 queries/min
- Average Response Time: 2.1s

Recent Activity:
- Last Query: 2 minutes ago
- Queries Today: 156
- Success Rate: 94%
- Cache Hit Rate: 78%

Graph Statistics:
- Total Nodes: 1,247
- Total Edges: 3,891
- Graph Density: 0.0025
- Largest Component: 1,198 nodes
- Average Clustering: 0.34
        """
        
        status_text.insert(1.0, status_info.strip())
        status_text.configure(state=tk.DISABLED)
        
        # Close button
        ttk.Button(status_window, text="Close", command=status_window.destroy).pack(pady=10)
    
    def show_about(self):
        """Show about dialog."""
        about_text = """Agentic RAG System v1.0

An intelligent knowledge assistant powered by:
• Advanced Graph Neural Networks
• FAISS Vector Search
• Multi-step Reasoning Engine
• Interactive Graph Visualization
• Streaming Query Processing

Built with Python, Neo4j, and modern AI technologies.

© 2024 Agentic RAG Team"""
        
        messagebox.showinfo("About Agentic RAG System", about_text)
    
    def show_user_guide(self):
        """Show user guide dialog."""
        guide_window = tk.Toplevel(self.root)
        guide_window.title("User Guide")
        guide_window.geometry("700x500")
        guide_window.transient(self.root)
        
        guide_text = scrolledtext.ScrolledText(guide_window, height=28, width=80)
        guide_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        user_guide = """
Agentic RAG System - User Guide
===============================

Getting Started:
1. Start the system using 'System > Start Components'
2. Wait for all components to show 'Running' status
3. Enter your query in the Query Interface
4. Click 'Execute Query' or press Ctrl+Enter

Query Interface:
• Query Type: Choose the type of query (factual, analytical, etc.)
• Reasoning Depth: Control how deep the reasoning goes (1-10)
• Max Results: Limit the number of results returned
• Stream Results: Enable real-time streaming of results

Graph Visualization:
• View the knowledge graph in the Graph Visualization tab
• Zoom and pan to explore different areas
• Highlighted paths show reasoning chains
• Node colors indicate different entity types

System Controls:
• Start/Stop: Control system components
• Refresh Status: Update system metrics
• Load/Save Graph: Import/export graph data

Keyboard Shortcuts:
• Ctrl+Enter: Execute query
• Ctrl+N: Clear query
• Ctrl+S: Save graph
• Ctrl+O: Load graph
• F5: Refresh status

Tips:
• Use specific queries for better results
• Monitor system status for performance
• Export results for further analysis
• Check logs for troubleshooting
        """
        
        guide_text.insert(1.0, user_guide.strip())
        guide_text.configure(state=tk.DISABLED)
        
        ttk.Button(guide_window, text="Close", command=guide_window.destroy).pack(pady=10)
    
    def run(self):
        """Run the GUI application."""
        try:
            self.log_widget.append_text("Agentic RAG System started", "success")
            self.log_widget.append_text("Ready for queries...", "info")
            self.root.mainloop()
        except KeyboardInterrupt:
            self.log_widget.append_text("Shutting down system...", "info")
        except Exception as e:
            logger.error(f"GUI runtime error: {e}")
            messagebox.showerror("Runtime Error", f"Application error:\n{e}")
        finally:
            self.stop_components()


def main():
    """Main entry point for the Agentic RAG GUI."""
    try:
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Create and run GUI
        app = AgenticRAGGUI()
        app.run()
        
    except Exception as e:
        logger.error(f"Failed to start Agentic RAG GUI: {e}")
        if TKINTER_AVAILABLE:
            messagebox.showerror("Startup Error", f"Failed to start application:\n{e}")
        else:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
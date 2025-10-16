#!/usr/bin/env python3
"""
Interactive Graph Editor for Agentic Graph RAG
Visual editor for continuous graph refinement and knowledge base improvement.
"""

import logging
import json
import asyncio
from typing import List, Dict, Any, Optional, Tuple, Set, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import uuid

try:
    import networkx as nx
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logging.warning("Plotly not available. Install with: pip install plotly networkx")

try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox, simpledialog
    import tkinter.scrolledtext as scrolledtext
    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False
    logging.warning("Tkinter not available for desktop GUI")

logger = logging.getLogger(__name__)


@dataclass
class GraphNode:
    """Represents a node in the knowledge graph."""
    id: str
    label: str
    node_type: str
    properties: Dict[str, Any] = field(default_factory=dict)
    x: Optional[float] = None
    y: Optional[float] = None
    z: Optional[float] = None
    color: str = "#3498db"
    size: int = 10
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'label': self.label,
            'type': self.node_type,
            'properties': self.properties,
            'position': {'x': self.x, 'y': self.y, 'z': self.z},
            'style': {'color': self.color, 'size': self.size}
        }


@dataclass
class GraphEdge:
    """Represents an edge in the knowledge graph."""
    id: str
    source: str
    target: str
    relationship: str
    properties: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0
    color: str = "#95a5a6"
    width: int = 2
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'source': self.source,
            'target': self.target,
            'relationship': self.relationship,
            'properties': self.properties,
            'weight': self.weight,
            'style': {'color': self.color, 'width': self.width}
        }


@dataclass
class GraphOperation:
    """Represents a graph modification operation."""
    operation_id: str
    operation_type: str  # add_node, add_edge, update_node, delete_node, etc.
    timestamp: datetime
    data: Dict[str, Any]
    user: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'operation_id': self.operation_id,
            'operation_type': self.operation_type,
            'timestamp': self.timestamp.isoformat(),
            'data': self.data,
            'user': self.user
        }


class GraphEditor:
    """
    Core graph editing functionality.
    
    Features:
    - Node and edge CRUD operations
    - Graph layout algorithms
    - Undo/redo functionality
    - Graph validation
    - Import/export capabilities
    - Change tracking and history
    """
    
    def __init__(self):
        """Initialize graph editor."""
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: Dict[str, GraphEdge] = {}
        self.graph = nx.DiGraph()
        
        # Operation history for undo/redo
        self.operation_history: List[GraphOperation] = []
        self.history_index = -1
        self.max_history = 100
        
        # Change callbacks
        self.change_callbacks: List[Callable] = []
        
        logger.info("Graph Editor initialized")
    
    def add_change_callback(self, callback: Callable):
        """Add callback for graph changes."""
        self.change_callbacks.append(callback)
    
    def _notify_change(self, operation: GraphOperation):
        """Notify all callbacks of graph change."""
        for callback in self.change_callbacks:
            try:
                callback(operation)
            except Exception as e:
                logger.error(f"Change callback failed: {e}")
    
    def _add_to_history(self, operation: GraphOperation):
        """Add operation to history."""
        # Remove future operations if we're not at the end
        if self.history_index < len(self.operation_history) - 1:
            self.operation_history = self.operation_history[:self.history_index + 1]
        
        # Add new operation
        self.operation_history.append(operation)
        self.history_index = len(self.operation_history) - 1
        
        # Limit history size
        if len(self.operation_history) > self.max_history:
            self.operation_history.pop(0)
            self.history_index -= 1
        
        # Notify change
        self._notify_change(operation)
    
    def add_node(self, 
                node_id: Optional[str] = None,
                label: str = "New Node",
                node_type: str = "Entity",
                properties: Optional[Dict[str, Any]] = None,
                position: Optional[Tuple[float, float, float]] = None) -> str:
        """Add a new node to the graph."""
        if node_id is None:
            node_id = f"node_{uuid.uuid4().hex[:8]}"
        
        if node_id in self.nodes:
            raise ValueError(f"Node {node_id} already exists")
        
        if properties is None:
            properties = {}
        
        # Create node
        node = GraphNode(
            id=node_id,
            label=label,
            node_type=node_type,
            properties=properties
        )
        
        # Set position if provided
        if position:
            node.x, node.y, node.z = position
        
        # Add to collections
        self.nodes[node_id] = node
        self.graph.add_node(node_id, **properties)
        
        # Record operation
        operation = GraphOperation(
            operation_id=f"add_node_{uuid.uuid4().hex[:8]}",
            operation_type="add_node",
            timestamp=datetime.now(),
            data={'node_id': node_id, 'node_data': node.to_dict()}
        )
        self._add_to_history(operation)
        
        logger.debug(f"Added node: {node_id}")
        return node_id
    
    def add_edge(self,
                source: str,
                target: str,
                relationship: str = "related_to",
                edge_id: Optional[str] = None,
                properties: Optional[Dict[str, Any]] = None,
                weight: float = 1.0) -> str:
        """Add a new edge to the graph."""
        if source not in self.nodes:
            raise ValueError(f"Source node {source} does not exist")
        if target not in self.nodes:
            raise ValueError(f"Target node {target} does not exist")
        
        if edge_id is None:
            edge_id = f"edge_{uuid.uuid4().hex[:8]}"
        
        if edge_id in self.edges:
            raise ValueError(f"Edge {edge_id} already exists")
        
        if properties is None:
            properties = {}
        
        # Create edge
        edge = GraphEdge(
            id=edge_id,
            source=source,
            target=target,
            relationship=relationship,
            properties=properties,
            weight=weight
        )
        
        # Add to collections
        self.edges[edge_id] = edge
        self.graph.add_edge(source, target, edge_id=edge_id, weight=weight, **properties)
        
        # Record operation
        operation = GraphOperation(
            operation_id=f"add_edge_{uuid.uuid4().hex[:8]}",
            operation_type="add_edge",
            timestamp=datetime.now(),
            data={'edge_id': edge_id, 'edge_data': edge.to_dict()}
        )
        self._add_to_history(operation)
        
        logger.debug(f"Added edge: {source} -> {target}")
        return edge_id
    
    def update_node(self, 
                   node_id: str, 
                   label: Optional[str] = None,
                   node_type: Optional[str] = None,
                   properties: Optional[Dict[str, Any]] = None,
                   position: Optional[Tuple[float, float, float]] = None):
        """Update an existing node."""
        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} does not exist")
        
        node = self.nodes[node_id]
        old_data = node.to_dict()
        
        # Update fields
        if label is not None:
            node.label = label
        if node_type is not None:
            node.node_type = node_type
        if properties is not None:
            node.properties.update(properties)
        if position is not None:
            node.x, node.y, node.z = position
        
        # Update NetworkX graph
        self.graph.nodes[node_id].update(node.properties)
        
        # Record operation
        operation = GraphOperation(
            operation_id=f"update_node_{uuid.uuid4().hex[:8]}",
            operation_type="update_node",
            timestamp=datetime.now(),
            data={
                'node_id': node_id,
                'old_data': old_data,
                'new_data': node.to_dict()
            }
        )
        self._add_to_history(operation)
        
        logger.debug(f"Updated node: {node_id}")
    
    def update_edge(self,
                   edge_id: str,
                   relationship: Optional[str] = None,
                   properties: Optional[Dict[str, Any]] = None,
                   weight: Optional[float] = None):
        """Update an existing edge."""
        if edge_id not in self.edges:
            raise ValueError(f"Edge {edge_id} does not exist")
        
        edge = self.edges[edge_id]
        old_data = edge.to_dict()
        
        # Update fields
        if relationship is not None:
            edge.relationship = relationship
        if properties is not None:
            edge.properties.update(properties)
        if weight is not None:
            edge.weight = weight
        
        # Update NetworkX graph
        edge_data = self.graph.get_edge_data(edge.source, edge.target)
        if edge_data:
            edge_data.update(edge.properties)
            if weight is not None:
                edge_data['weight'] = weight
        
        # Record operation
        operation = GraphOperation(
            operation_id=f"update_edge_{uuid.uuid4().hex[:8]}",
            operation_type="update_edge",
            timestamp=datetime.now(),
            data={
                'edge_id': edge_id,
                'old_data': old_data,
                'new_data': edge.to_dict()
            }
        )
        self._add_to_history(operation)
        
        logger.debug(f"Updated edge: {edge_id}")
    
    def delete_node(self, node_id: str):
        """Delete a node and all connected edges."""
        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} does not exist")
        
        # Find connected edges
        connected_edges = []
        for edge_id, edge in self.edges.items():
            if edge.source == node_id or edge.target == node_id:
                connected_edges.append(edge_id)
        
        # Store data for undo
        node_data = self.nodes[node_id].to_dict()
        edge_data = [self.edges[eid].to_dict() for eid in connected_edges]
        
        # Delete edges first
        for edge_id in connected_edges:
            del self.edges[edge_id]
        
        # Delete node
        del self.nodes[node_id]
        self.graph.remove_node(node_id)
        
        # Record operation
        operation = GraphOperation(
            operation_id=f"delete_node_{uuid.uuid4().hex[:8]}",
            operation_type="delete_node",
            timestamp=datetime.now(),
            data={
                'node_id': node_id,
                'node_data': node_data,
                'deleted_edges': edge_data
            }
        )
        self._add_to_history(operation)
        
        logger.debug(f"Deleted node: {node_id}")
    
    def delete_edge(self, edge_id: str):
        """Delete an edge."""
        if edge_id not in self.edges:
            raise ValueError(f"Edge {edge_id} does not exist")
        
        edge = self.edges[edge_id]
        edge_data = edge.to_dict()
        
        # Remove from NetworkX graph
        if self.graph.has_edge(edge.source, edge.target):
            self.graph.remove_edge(edge.source, edge.target)
        
        # Remove from collection
        del self.edges[edge_id]
        
        # Record operation
        operation = GraphOperation(
            operation_id=f"delete_edge_{uuid.uuid4().hex[:8]}",
            operation_type="delete_edge",
            timestamp=datetime.now(),
            data={'edge_id': edge_id, 'edge_data': edge_data}
        )
        self._add_to_history(operation)
        
        logger.debug(f"Deleted edge: {edge_id}")
    
    def undo(self) -> bool:
        """Undo the last operation."""
        if self.history_index < 0:
            return False
        
        operation = self.operation_history[self.history_index]
        
        # Reverse the operation
        try:
            if operation.operation_type == "add_node":
                node_id = operation.data['node_id']
                if node_id in self.nodes:
                    # Remove without recording history
                    del self.nodes[node_id]
                    if self.graph.has_node(node_id):
                        self.graph.remove_node(node_id)
            
            elif operation.operation_type == "add_edge":
                edge_id = operation.data['edge_id']
                if edge_id in self.edges:
                    edge = self.edges[edge_id]
                    del self.edges[edge_id]
                    if self.graph.has_edge(edge.source, edge.target):
                        self.graph.remove_edge(edge.source, edge.target)
            
            elif operation.operation_type == "delete_node":
                # Restore node and edges
                node_data = operation.data['node_data']
                node = GraphNode(**node_data)
                self.nodes[node.id] = node
                self.graph.add_node(node.id, **node.properties)
                
                # Restore edges
                for edge_data in operation.data.get('deleted_edges', []):
                    edge = GraphEdge(**edge_data)
                    self.edges[edge.id] = edge
                    self.graph.add_edge(edge.source, edge.target, **edge.properties)
            
            # Add more operation reversals as needed...
            
            self.history_index -= 1
            logger.debug(f"Undid operation: {operation.operation_type}")
            return True
        
        except Exception as e:
            logger.error(f"Undo failed: {e}")
            return False
    
    def redo(self) -> bool:
        """Redo the next operation."""
        if self.history_index >= len(self.operation_history) - 1:
            return False
        
        self.history_index += 1
        operation = self.operation_history[self.history_index]
        
        # Reapply the operation
        try:
            if operation.operation_type == "add_node":
                node_data = operation.data['node_data']
                node = GraphNode(**node_data)
                self.nodes[node.id] = node
                self.graph.add_node(node.id, **node.properties)
            
            # Add more operation reapplications as needed...
            
            logger.debug(f"Redid operation: {operation.operation_type}")
            return True
        
        except Exception as e:
            logger.error(f"Redo failed: {e}")
            return False
    
    def apply_layout(self, layout_type: str = "spring") -> Dict[str, Tuple[float, float]]:
        """Apply layout algorithm to position nodes."""
        if not self.nodes:
            return {}
        
        try:
            if layout_type == "spring":
                pos = nx.spring_layout(self.graph, k=1, iterations=50)
            elif layout_type == "circular":
                pos = nx.circular_layout(self.graph)
            elif layout_type == "random":
                pos = nx.random_layout(self.graph)
            elif layout_type == "shell":
                pos = nx.shell_layout(self.graph)
            else:
                pos = nx.spring_layout(self.graph)
            
            # Update node positions
            for node_id, (x, y) in pos.items():
                if node_id in self.nodes:
                    self.nodes[node_id].x = float(x)
                    self.nodes[node_id].y = float(y)
                    self.nodes[node_id].z = 0.0
            
            logger.debug(f"Applied {layout_type} layout to {len(pos)} nodes")
            return pos
        
        except Exception as e:
            logger.error(f"Layout application failed: {e}")
            return {}
    
    def export_graph(self, format_type: str = "json") -> str:
        """Export graph in various formats."""
        try:
            if format_type == "json":
                graph_data = {
                    'nodes': [node.to_dict() for node in self.nodes.values()],
                    'edges': [edge.to_dict() for edge in self.edges.values()],
                    'metadata': {
                        'node_count': len(self.nodes),
                        'edge_count': len(self.edges),
                        'exported_at': datetime.now().isoformat()
                    }
                }
                return json.dumps(graph_data, indent=2)
            
            elif format_type == "gexf":
                # NetworkX GEXF export
                return '\n'.join(nx.generate_gexf(self.graph))
            
            else:
                raise ValueError(f"Unsupported export format: {format_type}")
        
        except Exception as e:
            logger.error(f"Export failed: {e}")
            raise
    
    def import_graph(self, data: str, format_type: str = "json", merge: bool = False):
        """Import graph from various formats."""
        try:
            if not merge:
                # Clear existing graph
                self.nodes.clear()
                self.edges.clear()
                self.graph.clear()
            
            if format_type == "json":
                graph_data = json.loads(data)
                
                # Import nodes
                for node_data in graph_data.get('nodes', []):
                    node = GraphNode(
                        id=node_data['id'],
                        label=node_data['label'],
                        node_type=node_data['type'],
                        properties=node_data.get('properties', {})
                    )
                    
                    # Set position if available
                    pos = node_data.get('position', {})
                    if pos:
                        node.x = pos.get('x')
                        node.y = pos.get('y')
                        node.z = pos.get('z')
                    
                    # Set style if available
                    style = node_data.get('style', {})
                    if style:
                        node.color = style.get('color', node.color)
                        node.size = style.get('size', node.size)
                    
                    self.nodes[node.id] = node
                    self.graph.add_node(node.id, **node.properties)
                
                # Import edges
                for edge_data in graph_data.get('edges', []):
                    edge = GraphEdge(
                        id=edge_data['id'],
                        source=edge_data['source'],
                        target=edge_data['target'],
                        relationship=edge_data['relationship'],
                        properties=edge_data.get('properties', {}),
                        weight=edge_data.get('weight', 1.0)
                    )
                    
                    # Set style if available
                    style = edge_data.get('style', {})
                    if style:
                        edge.color = style.get('color', edge.color)
                        edge.width = style.get('width', edge.width)
                    
                    self.edges[edge.id] = edge
                    
                    # Only add edge if both nodes exist
                    if edge.source in self.nodes and edge.target in self.nodes:
                        self.graph.add_edge(edge.source, edge.target, **edge.properties)
            
            logger.info(f"Imported {len(self.nodes)} nodes and {len(self.edges)} edges")
        
        except Exception as e:
            logger.error(f"Import failed: {e}")
            raise
    
    def validate_graph(self) -> List[str]:
        """Validate graph consistency and return issues."""
        issues = []
        
        # Check for orphaned edges
        for edge_id, edge in self.edges.items():
            if edge.source not in self.nodes:
                issues.append(f"Edge {edge_id} references non-existent source node {edge.source}")
            if edge.target not in self.nodes:
                issues.append(f"Edge {edge_id} references non-existent target node {edge.target}")
        
        # Check for isolated nodes
        isolated = list(nx.isolates(self.graph))
        if isolated:
            issues.append(f"Found {len(isolated)} isolated nodes: {isolated[:5]}...")
        
        # Check for self-loops
        self_loops = list(nx.nodes_with_selfloops(self.graph))
        if self_loops:
            issues.append(f"Found {len(self_loops)} nodes with self-loops")
        
        return issues
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics."""
        stats = {
            'nodes': len(self.nodes),
            'edges': len(self.edges),
            'density': nx.density(self.graph) if self.nodes else 0,
            'is_connected': nx.is_connected(self.graph.to_undirected()) if self.nodes else False,
            'average_degree': sum(dict(self.graph.degree()).values()) / len(self.nodes) if self.nodes else 0
        }
        
        # Node type distribution
        node_types = {}
        for node in self.nodes.values():
            node_types[node.node_type] = node_types.get(node.node_type, 0) + 1
        stats['node_types'] = node_types
        
        # Relationship distribution
        relationships = {}
        for edge in self.edges.values():
            relationships[edge.relationship] = relationships.get(edge.relationship, 0) + 1
        stats['relationships'] = relationships
        
        return stats


class InteractiveGraphEditor:
    """
    Interactive graph editor with visualization capabilities.
    
    Features:
    - Real-time graph visualization
    - Interactive node/edge manipulation
    - Multiple layout algorithms
    - Search and filtering
    - Export/import functionality
    - Collaborative editing support
    """
    
    def __init__(self, use_plotly: bool = True):
        """Initialize interactive graph editor."""
        self.graph_editor = GraphEditor()
        self.use_plotly = use_plotly and PLOTLY_AVAILABLE
        
        # Visualization settings
        self.layout_type = "spring"
        self.node_color_mapping = {
            "Entity": "#3498db",
            "Concept": "#e74c3c", 
            "Event": "#f39c12",
            "Location": "#27ae60",
            "Person": "#9b59b6"
        }
        
        # Selection state
        self.selected_nodes: Set[str] = set()
        self.selected_edges: Set[str] = set()
        
        # Filter state
        self.node_filters: Dict[str, Any] = {}
        self.edge_filters: Dict[str, Any] = {}
        
        logger.info("Interactive Graph Editor initialized")
    
    def create_visualization(self, 
                           width: int = 800, 
                           height: int = 600,
                           layout_3d: bool = False) -> Optional[go.Figure]:
        """Create interactive visualization of the graph."""
        if not self.use_plotly:
            logger.warning("Plotly not available for visualization")
            return None
        
        if not self.graph_editor.nodes:
            logger.info("No nodes to visualize")
            return None
        
        # Apply layout if positions not set
        if not any(node.x is not None for node in self.graph_editor.nodes.values()):
            self.graph_editor.apply_layout(self.layout_type)
        
        fig = go.Figure()
        
        # Add edges
        edge_x, edge_y, edge_z = [], [], []
        edge_info = []
        
        for edge in self.graph_editor.edges.values():
            source_node = self.graph_editor.nodes[edge.source]
            target_node = self.graph_editor.nodes[edge.target]
            
            # Edge coordinates
            edge_x.extend([source_node.x, target_node.x, None])
            edge_y.extend([source_node.y, target_node.y, None])
            edge_z.extend([source_node.z or 0, target_node.z or 0, None])
            
            edge_info.append(f"{edge.source} → {edge.target}<br>{edge.relationship}")
        
        # Create edge trace
        if layout_3d:
            edge_trace = go.Scatter3d(
                x=edge_x, y=edge_y, z=edge_z,
                line=dict(width=2, color='#888'),
                hoverinfo='none',
                mode='lines'
            )
        else:
            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=2, color='#888'),
                hoverinfo='none',
                mode='lines'
            )
        
        fig.add_trace(edge_trace)
        
        # Add nodes
        node_x, node_y, node_z = [], [], []
        node_text, node_color, node_size = [], [], []
        node_hover = []
        
        for node in self.graph_editor.nodes.values():
            node_x.append(node.x)
            node_y.append(node.y)
            node_z.append(node.z or 0)
            node_text.append(node.label)
            node_color.append(self.node_color_mapping.get(node.node_type, "#3498db"))
            node_size.append(node.size)
            
            # Hover information
            properties_str = "<br>".join([f"{k}: {v}" for k, v in node.properties.items()])
            hover_text = f"<b>{node.label}</b><br>Type: {node.node_type}<br>ID: {node.id}"
            if properties_str:
                hover_text += f"<br><br>Properties:<br>{properties_str}"
            node_hover.append(hover_text)
        
        # Create node trace
        if layout_3d:
            node_trace = go.Scatter3d(
                x=node_x, y=node_y, z=node_z,
                mode='markers+text',
                marker=dict(
                    size=node_size,
                    color=node_color,
                    line=dict(width=2, color='white')
                ),
                text=node_text,
                textposition="middle center",
                hoverinfo='text',
                hovertext=node_hover
            )
        else:
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                marker=dict(
                    size=node_size,
                    color=node_color,
                    line=dict(width=2, color='white')
                ),
                text=node_text,
                textposition="middle center",
                hoverinfo='text',
                hovertext=node_hover
            )
        
        fig.add_trace(node_trace)
        
        # Configure layout
        if layout_3d:
            fig.update_layout(
                title="Interactive Knowledge Graph",
                showlegend=False,
                scene=dict(
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    zaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                ),
                width=width,
                height=height
            )
        else:
            fig.update_layout(
                title="Interactive Knowledge Graph",
                showlegend=False,
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                width=width,
                height=height,
                plot_bgcolor='white'
            )
        
        return fig
    
    def save_visualization(self, 
                         filename: str,
                         format_type: str = "html",
                         width: int = 800,
                         height: int = 600):
        """Save visualization to file."""
        fig = self.create_visualization(width, height)
        if fig is None:
            raise ValueError("Cannot create visualization")
        
        if format_type == "html":
            fig.write_html(filename)
        elif format_type == "png":
            fig.write_image(filename)
        elif format_type == "svg":
            fig.write_image(filename)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
        
        logger.info(f"Visualization saved to {filename}")
    
    def search_nodes(self, query: str) -> List[str]:
        """Search nodes by label or properties."""
        results = []
        query_lower = query.lower()
        
        for node_id, node in self.graph_editor.nodes.items():
            # Search in label
            if query_lower in node.label.lower():
                results.append(node_id)
                continue
            
            # Search in properties
            for value in node.properties.values():
                if isinstance(value, str) and query_lower in value.lower():
                    results.append(node_id)
                    break
        
        return results
    
    def filter_graph(self, 
                    node_types: Optional[List[str]] = None,
                    relationships: Optional[List[str]] = None,
                    property_filters: Optional[Dict[str, Any]] = None):
        """Apply filters to graph visualization."""
        # Store filter state
        if node_types is not None:
            self.node_filters['types'] = node_types
        if relationships is not None:
            self.edge_filters['relationships'] = relationships
        if property_filters is not None:
            self.node_filters['properties'] = property_filters
        
        # Implementation would filter the visualization
        # This is a placeholder for the filtering logic
        logger.debug(f"Applied filters: nodes={node_types}, edges={relationships}")
    
    def clear_filters(self):
        """Clear all filters."""
        self.node_filters.clear()
        self.edge_filters.clear()
        logger.debug("Cleared all filters")
    
    def select_nodes(self, node_ids: List[str]):
        """Select nodes for operations."""
        self.selected_nodes.update(node_ids)
        logger.debug(f"Selected {len(node_ids)} nodes")
    
    def select_edges(self, edge_ids: List[str]):
        """Select edges for operations."""
        self.selected_edges.update(edge_ids)
        logger.debug(f"Selected {len(edge_ids)} edges")
    
    def clear_selection(self):
        """Clear all selections."""
        self.selected_nodes.clear()
        self.selected_edges.clear()
        logger.debug("Cleared selection")
    
    def group_selected_nodes(self, group_name: str) -> str:
        """Group selected nodes."""
        if not self.selected_nodes:
            raise ValueError("No nodes selected")
        
        # Create group node
        group_id = self.graph_editor.add_node(
            label=group_name,
            node_type="Group",
            properties={'members': list(self.selected_nodes)}
        )
        
        # Connect group to members
        for node_id in self.selected_nodes:
            self.graph_editor.add_edge(
                group_id, node_id,
                relationship="contains"
            )
        
        logger.info(f"Created group {group_name} with {len(self.selected_nodes)} members")
        return group_id
    
    def apply_layout_algorithm(self, algorithm: str):
        """Apply layout algorithm and update visualization."""
        self.layout_type = algorithm
        self.graph_editor.apply_layout(algorithm)
        logger.info(f"Applied {algorithm} layout")
    
    def export_selection(self) -> Dict[str, Any]:
        """Export selected nodes and edges."""
        selected_data = {
            'nodes': [],
            'edges': []
        }
        
        # Export selected nodes
        for node_id in self.selected_nodes:
            if node_id in self.graph_editor.nodes:
                selected_data['nodes'].append(
                    self.graph_editor.nodes[node_id].to_dict()
                )
        
        # Export selected edges
        for edge_id in self.selected_edges:
            if edge_id in self.graph_editor.edges:
                selected_data['edges'].append(
                    self.graph_editor.edges[edge_id].to_dict()
                )
        
        return selected_data
    
    def get_node_recommendations(self, node_id: str) -> List[Dict[str, Any]]:
        """Get recommendations for potential connections."""
        if node_id not in self.graph_editor.nodes:
            return []
        
        recommendations = []
        node = self.graph_editor.nodes[node_id]
        
        # Find nodes with similar properties
        for other_id, other_node in self.graph_editor.nodes.items():
            if other_id == node_id:
                continue
            
            # Calculate similarity based on properties
            similarity_score = 0
            common_props = set(node.properties.keys()) & set(other_node.properties.keys())
            
            for prop in common_props:
                if node.properties[prop] == other_node.properties[prop]:
                    similarity_score += 1
            
            if similarity_score > 0:
                recommendations.append({
                    'node_id': other_id,
                    'label': other_node.label,
                    'similarity_score': similarity_score,
                    'reason': f"Shares {similarity_score} common properties"
                })
        
        # Sort by similarity
        recommendations.sort(key=lambda x: x['similarity_score'], reverse=True)
        return recommendations[:10]  # Top 10 recommendations


# Convenience functions
def create_graph_editor() -> InteractiveGraphEditor:
    """Create interactive graph editor."""
    return InteractiveGraphEditor()


def load_graph_from_file(filename: str) -> InteractiveGraphEditor:
    """Load graph from file."""
    editor = InteractiveGraphEditor()
    
    with open(filename, 'r', encoding='utf-8') as f:
        data = f.read()
    
    editor.graph_editor.import_graph(data, "json")
    return editor
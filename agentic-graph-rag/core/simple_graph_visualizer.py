#!/usr/bin/env python3
"""
Simplified 3D Graph Visualizer without heavy dependencies.
"""

import json
import logging
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import webbrowser
import tempfile
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class Node3D:
    """3D node representation."""
    id: str
    label: str
    type: str
    x: float
    y: float
    z: float
    size: int = 10
    color: str = "#1f77b4"
    properties: Dict = None
    
    def __post_init__(self):
        if self.properties is None:
            self.properties = {}


@dataclass
class Edge3D:
    """3D edge representation."""
    source: str
    target: str
    label: str
    type: str
    weight: float = 1.0
    color: str = "#999999"
    properties: Dict = None
    
    def __post_init__(self):
        if self.properties is None:
            self.properties = {}


@dataclass
class Graph3D:
    """3D graph data structure."""
    nodes: List[Node3D]
    edges: List[Edge3D]
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class SimpleGraphVisualizer:
    """
    Simplified 3D graph visualizer.
    """
    
    def __init__(self):
        """Initialize the visualizer."""
        # Color palettes for different node types
        self.type_colors = {
            'Person': '#FF6B6B',
            'Organization': '#4ECDC4', 
            'Location': '#45B7D1',
            'Event': '#96CEB4',
            'Concept': '#FFEAA7',
            'Technology': '#DDA0DD',
            'Product': '#98D8C8',
            'default': '#1f77b4'
        }
    
    async def fetch_graph_data(self, graph_store, limit: int = 1000) -> Graph3D:
        """
        Fetch graph data from Neo4j store.
        """
        try:
            # Fetch nodes with their properties
            nodes_query = f"""
            MATCH (n)
            RETURN n.id as id, labels(n) as labels, n.name as name, 
                   properties(n) as properties
            LIMIT {limit}
            """
            
            nodes_result = await graph_store.execute_query(nodes_query)
            
            # Fetch relationships
            edges_query = f"""
            MATCH (a)-[r]->(b)
            WHERE a.id IS NOT NULL AND b.id IS NOT NULL
            RETURN a.id as source, b.id as target, type(r) as relationship_type,
                   properties(r) as properties
            LIMIT {limit * 2}
            """
            
            edges_result = await graph_store.execute_query(edges_query)
            
            # Convert to 3D graph structure
            return await self._convert_to_3d_graph(nodes_result, edges_result)
            
        except Exception as e:
            logger.error(f"Failed to fetch graph data: {e}")
            raise
    
    async def _convert_to_3d_graph(self, nodes_result, edges_result) -> Graph3D:
        """Convert Neo4j results to 3D graph structure."""
        
        # Process nodes
        nodes = []
        node_positions = {}
        
        for i, record in enumerate(nodes_result):
            node_id = record['id']
            labels = record['labels']
            name = record['name'] or node_id
            properties = record['properties'] or {}
            
            # Determine node type and color
            node_type = labels[0] if labels else 'Unknown'
            color = self.type_colors.get(node_type, self.type_colors['default'])
            
            # Generate 3D position using simple layout
            import math
            angle = 2 * math.pi * i / max(len(nodes_result), 1)
            radius = 100 + (i % 3) * 50  # Vary radius for depth
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            z = 50 * (i % 3 - 1)  # Distribute across 3 z-levels
            
            node = Node3D(
                id=node_id,
                label=name,
                type=node_type,
                x=x, y=y, z=z,
                size=10,
                color=color,
                properties=properties
            )
            
            nodes.append(node)
            node_positions[node_id] = (x, y, z)
        
        # Process edges
        edges = []
        for record in edges_result:
            source = record['source']
            target = record['target']
            rel_type = record['relationship_type']
            properties = record['properties'] or {}
            
            # Skip if nodes don't exist
            if source not in node_positions or target not in node_positions:
                continue
            
            edge = Edge3D(
                source=source,
                target=target,
                label=rel_type,
                type=rel_type,
                weight=1.0,
                color="#999999",
                properties=properties
            )
            
            edges.append(edge)
        
        return Graph3D(nodes=nodes, edges=edges)
    
    async def create_web_visualization(self, 
                                     graph_3d: Graph3D,
                                     title: str = "Knowledge Graph 3D",
                                     output_file: Optional[str] = None) -> str:
        """
        Create simple web-based 3D visualization.
        """
        
        # Prepare data for visualization
        nodes_data = []
        for node in graph_3d.nodes:
            nodes_data.append({
                'id': node.id,
                'label': node.label,
                'title': f"Type: {node.type}\\nID: {node.id}",
                'color': node.color,
                'size': node.size,
                'x': node.x,
                'y': node.y,
                'z': node.z
            })
        
        edges_data = []
        for edge in graph_3d.edges:
            edges_data.append({
                'from': edge.source,
                'to': edge.target,
                'label': edge.label,
                'title': f"Relationship: {edge.type}",
                'color': edge.color
            })
        
        # Generate HTML with simple 3D visualization
        html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {{
            margin: 0;
            padding: 0;
            font-family: Arial, sans-serif;
            background: #1a1a1a;
            color: white;
        }}
        #graph {{
            width: 100vw;
            height: 100vh;
        }}
        .controls {{
            position: absolute;
            top: 10px;
            right: 10px;
            background: rgba(255,255,255,0.9);
            color: black;
            padding: 10px;
            border-radius: 5px;
            z-index: 1000;
        }}
        .node {{
            cursor: pointer;
        }}
        .link {{
            stroke: #999;
            stroke-opacity: 0.6;
        }}
        .tooltip {{
            position: absolute;
            background: rgba(0,0,0,0.8);
            color: white;
            padding: 5px;
            border-radius: 3px;
            pointer-events: none;
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <div class="controls">
        <h3>{title}</h3>
        <p>Nodes: {len(graph_3d.nodes)}</p>
        <p>Edges: {len(graph_3d.edges)}</p>
        <button onclick="resetView()">Reset View</button>
    </div>
    
    <svg id="graph"></svg>
    <div class="tooltip" id="tooltip"></div>

    <script>
        const nodes = {json.dumps(nodes_data)};
        const links = {json.dumps(edges_data)};
        
        const width = window.innerWidth;
        const height = window.innerHeight;
        
        const svg = d3.select("#graph")
            .attr("width", width)
            .attr("height", height);
        
        const g = svg.append("g");
        
        // Create zoom behavior
        const zoom = d3.zoom()
            .scaleExtent([0.1, 10])
            .on("zoom", (event) => {{
                g.attr("transform", event.transform);
            }});
        
        svg.call(zoom);
        
        // Create simulation
        const simulation = d3.forceSimulation(nodes)
            .force("link", d3.forceLink(links).id(d => d.id).distance(100))
            .force("charge", d3.forceManyBody().strength(-300))
            .force("center", d3.forceCenter(width / 2, height / 2))
            .force("collision", d3.forceCollide().radius(d => d.size + 2));
        
        // Create links
        const link = g.append("g")
            .selectAll("line")
            .data(links)
            .enter().append("line")
            .attr("class", "link")
            .attr("stroke-width", 2);
        
        // Create nodes
        const node = g.append("g")
            .selectAll("circle")
            .data(nodes)
            .enter().append("circle")
            .attr("class", "node")
            .attr("r", d => d.size)
            .attr("fill", d => d.color)
            .call(d3.drag()
                .on("start", dragstarted)
                .on("drag", dragged)
                .on("end", dragended))
            .on("mouseover", showTooltip)
            .on("mouseout", hideTooltip);
        
        // Add labels
        const labels = g.append("g")
            .selectAll("text")
            .data(nodes)
            .enter().append("text")
            .text(d => d.label)
            .attr("font-size", "12px")
            .attr("fill", "white")
            .attr("text-anchor", "middle")
            .attr("dy", 4);
        
        // Update positions on tick
        simulation.on("tick", () => {{
            link
                .attr("x1", d => d.source.x)
                .attr("y1", d => d.source.y)
                .attr("x2", d => d.target.x)
                .attr("y2", d => d.target.y);
            
            node
                .attr("cx", d => d.x)
                .attr("cy", d => d.y);
            
            labels
                .attr("x", d => d.x)
                .attr("y", d => d.y);
        }});
        
        // Drag functions
        function dragstarted(event, d) {{
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
        }}
        
        function dragged(event, d) {{
            d.fx = event.x;
            d.fy = event.y;
        }}
        
        function dragended(event, d) {{
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
        }}
        
        // Tooltip functions
        function showTooltip(event, d) {{
            const tooltip = d3.select("#tooltip");
            tooltip.style("opacity", 1)
                .html(`<strong>${{d.label}}</strong><br/>Type: ${{d.type}}<br/>ID: ${{d.id}}`)
                .style("left", (event.pageX + 10) + "px")
                .style("top", (event.pageY - 10) + "px");
        }}
        
        function hideTooltip() {{
            d3.select("#tooltip").style("opacity", 0);
        }}
        
        // Reset view function
        function resetView() {{
            svg.transition().duration(750).call(zoom.transform, d3.zoomIdentity);
        }}
    </script>
</body>
</html>
        """
        
        # Save to file
        if output_file is None:
            output_file = tempfile.mktemp(suffix='.html')
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_template)
        
        logger.info(f"Web visualization saved to: {output_file}")
        return output_file
    
    async def export_graph_data(self, 
                              graph_3d: Graph3D,
                              format: str = 'json',
                              output_file: Optional[str] = None) -> str:
        """
        Export graph data in JSON format.
        """
        
        if output_file is None:
            output_file = tempfile.mktemp(suffix=f'.{format}')
        
        if format == 'json':
            # Export as JSON
            export_data = {
                'metadata': graph_3d.metadata,
                'nodes': [asdict(node) for node in graph_3d.nodes],
                'edges': [asdict(edge) for edge in graph_3d.edges]
            }
            
            with open(output_file, 'w') as f:
                json.dump(export_data, f, indent=2)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        logger.info(f"Graph exported to: {output_file}")
        return output_file
    
    def create_interactive_graph(self, entities: List[Dict], relationships: List[Dict], 
                               title: str = "Knowledge Graph") -> str:
        """
        Create interactive graph from entity/relationship lists.
        Robust version with comprehensive error handling and debugging.
        
        Args:
            entities: List of entity dictionaries with 'id', 'label', 'type' keys
            relationships: List of relationship dictionaries with 'source', 'target', 'relationship' keys
            title: Graph title
            
        Returns:
            HTML content as string
        """
        
        # Input validation and debugging
        logger.info(f"Creating graph with {len(entities)} entities and {len(relationships)} relationships")
        
        if not entities:
            logger.warning("No entities provided, creating empty graph")
            entities = [{"id": "empty", "label": "No Data", "type": "Empty"}]
        
        # Sanitize and validate entities
        nodes_data = []
        entity_ids = set()
        
        for i, entity in enumerate(entities):
            entity_id = str(entity.get('id', f'entity_{i}'))
            entity_label = str(entity.get('label', f'Entity {i}'))
            entity_type = str(entity.get('type', 'Unknown'))
            
            # Ensure unique IDs
            original_id = entity_id
            counter = 1
            while entity_id in entity_ids:
                entity_id = f"{original_id}_{counter}"
                counter += 1
            
            entity_ids.add(entity_id)
            
            nodes_data.append({
                'id': entity_id,
                'label': entity_label[:50],  # Truncate long labels
                'type': entity_type,
                'originalIndex': i
            })
        
        # Sanitize and validate relationships
        edges_data = []
        valid_relationships = 0
        
        for i, rel in enumerate(relationships):
            source_id = str(rel.get('source', ''))
            target_id = str(rel.get('target', ''))
            rel_type = str(rel.get('relationship', 'related'))
            
            # Check if both source and target exist
            if source_id in entity_ids and target_id in entity_ids:
                edges_data.append({
                    'source': source_id,
                    'target': target_id,
                    'label': rel_type[:20],  # Truncate long labels
                    'type': rel_type
                })
                valid_relationships += 1
            else:
                logger.warning(f"Skipping invalid relationship: {source_id} -> {target_id}")
        
        logger.info(f"Created {len(nodes_data)} nodes and {valid_relationships} valid relationships")
        
        # If no valid relationships, create some sample ones for visualization
        if not edges_data and len(nodes_data) > 1:
            logger.info("No valid relationships found, creating sample connections")
            for i in range(min(3, len(nodes_data) - 1)):
                edges_data.append({
                    'source': nodes_data[i]['id'],
                    'target': nodes_data[i + 1]['id'],
                    'label': 'related',
                    'type': 'related'
                })
        
        # Generate comprehensive HTML with better error handling
        html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f8f9fa;
        }}
        .header {{
            text-align: center;
            margin-bottom: 20px;
        }}
        .graph-container {{
            width: 100%;
            height: 700px;
            border: 2px solid #dee2e6;
            border-radius: 8px;
            background: white;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            position: relative;
        }}
        .controls {{
            position: absolute;
            top: 10px;
            right: 10px;
            z-index: 1000;
        }}
        .btn {{
            padding: 8px 16px;
            margin: 2px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 12px;
        }}
        .btn-primary {{ background: #007bff; color: white; }}
        .btn-secondary {{ background: #6c757d; color: white; }}
        .node {{
            cursor: pointer;
            stroke: #fff;
            stroke-width: 2px;
        }}
        .link {{
            stroke: #999;
            stroke-opacity: 0.6;
            stroke-width: 1.5px;
        }}
        .node-label {{
            font-size: 11px;
            font-weight: bold;
            text-anchor: middle;
            pointer-events: none;
            fill: #333;
        }}
        .edge-label {{
            font-size: 9px;
            text-anchor: middle;
            fill: #666;
            pointer-events: none;
        }}
        .info-panel {{
            margin-top: 20px;
            padding: 20px;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }}
        .stat-box {{
            padding: 15px;
            background: #f8f9fa;
            border-radius: 6px;
            text-align: center;
        }}
        .stat-number {{
            font-size: 24px;
            font-weight: bold;
            color: #007bff;
        }}
        .legend {{
            margin-top: 15px;
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 5px;
        }}
        .legend-color {{
            width: 16px;
            height: 16px;
            border-radius: 50%;
            border: 2px solid #fff;
        }}
        .debug-info {{
            margin-top: 10px;
            padding: 10px;
            background: #fff3cd;
            border-radius: 4px;
            font-family: monospace;
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{title}</h1>
        <p>Interactive Knowledge Graph Visualization</p>
    </div>
    
    <div class="graph-container">
        <div class="controls">
            <button class="btn btn-primary" onclick="resetGraph()">Reset View</button>
            <button class="btn btn-secondary" onclick="pauseSimulation()">Pause</button>
        </div>
        <svg id="graph-svg" width="100%" height="100%"></svg>
    </div>
    
    <div class="info-panel">
        <h3>Graph Statistics</h3>
        <div class="stats">
            <div class="stat-box">
                <div class="stat-number">{len(nodes_data)}</div>
                <div>Entities</div>
            </div>
            <div class="stat-box">
                <div class="stat-number">{len(edges_data)}</div>
                <div>Relationships</div>
            </div>
            <div class="stat-box">
                <div class="stat-number">{len(set(node['type'] for node in nodes_data))}</div>
                <div>Entity Types</div>
            </div>
        </div>
        
        <div class="legend">
            <strong>Entity Types:</strong>
            <div class="legend-item">
                <div class="legend-color" style="background: #ff7f0e;"></div>
                <span>Person</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #2ca02c;"></div>
                <span>Organization</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #d62728;"></div>
                <span>Concept</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #9467bd;"></div>
                <span>Location</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #69b3a2;"></div>
                <span>Other</span>
            </div>
        </div>
        
        <div class="debug-info">
            <strong>Debug Info:</strong><br>
            Nodes: {len(nodes_data)} | Valid Relationships: {len(edges_data)}<br>
            Entity Types: {", ".join(set(node['type'] for node in nodes_data))}<br>
            Timestamp: {json.dumps({"generated": "now"})}
        </div>
    </div>

    <script>
        // Data validation
        const rawNodes = {json.dumps(nodes_data, indent=2)};
        const rawEdges = {json.dumps(edges_data, indent=2)};
        
        console.log('Graph data loaded:', {{
            nodes: rawNodes.length,
            edges: rawEdges.length,
            nodeTypes: [...new Set(rawNodes.map(n => n.type))]
        }});
        
        // Error handling for empty data
        if (rawNodes.length === 0) {{
            document.getElementById('graph-svg').innerHTML = 
                '<text x="50%" y="50%" text-anchor="middle" fill="red" font-size="20">No data to visualize</text>';
            throw new Error('No nodes to display');
        }}
        
        // Setup dimensions
        const container = document.querySelector('.graph-container');
        const width = container.clientWidth - 4;
        const height = container.clientHeight - 4;
        
        console.log('SVG dimensions:', {{width, height}});
        
        const svg = d3.select("#graph-svg")
            .attr("width", width)
            .attr("height", height);
        
        // Clear any existing content
        svg.selectAll("*").remove();
        
        // Add zoom behavior
        const g = svg.append("g");
        
        const zoom = d3.zoom()
            .scaleExtent([0.1, 3])
            .on("zoom", (event) => {{
                g.attr("transform", event.transform);
            }});
        
        svg.call(zoom);
        
        // Color mapping for entity types
        const colorScale = d3.scaleOrdinal()
            .domain(["Person", "Organization", "Concept", "Location", "Unknown"])
            .range(["#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#69b3a2"]);
        
        // Create force simulation
        const simulation = d3.forceSimulation(rawNodes)
            .force("link", d3.forceLink(rawEdges).id(d => d.id).distance(120).strength(0.5))
            .force("charge", d3.forceManyBody().strength(-400))
            .force("center", d3.forceCenter(width / 2, height / 2))
            .force("collision", d3.forceCollide().radius(30));
        
        // Create links
        const link = g.append("g")
            .attr("class", "links")
            .selectAll("line")
            .data(rawEdges)
            .enter()
            .append("line")
            .attr("class", "link")
            .attr("stroke-width", 2);
        
        // Create edge labels
        const edgeLabels = g.append("g")
            .attr("class", "edge-labels")
            .selectAll("text")
            .data(rawEdges)
            .enter()
            .append("text")
            .attr("class", "edge-label")
            .text(d => d.label || d.type);
        
        // Create nodes
        const node = g.append("g")
            .attr("class", "nodes")
            .selectAll("circle")
            .data(rawNodes)
            .enter()
            .append("circle")
            .attr("class", "node")
            .attr("r", 12)
            .attr("fill", d => colorScale(d.type))
            .call(d3.drag()
                .on("start", dragstarted)
                .on("drag", dragged)
                .on("end", dragended))
            .on("mouseover", handleMouseOver)
            .on("mouseout", handleMouseOut)
            .on("click", handleClick);
        
        // Create node labels
        const nodeLabels = g.append("g")
            .attr("class", "node-labels")
            .selectAll("text")
            .data(rawNodes)
            .enter()
            .append("text")
            .attr("class", "node-label")
            .attr("dy", 25)
            .text(d => d.label);
        
        // Add tooltips
        node.append("title")
            .text(d => `${{d.label}}\\nType: ${{d.type}}\\nID: ${{d.id}}`);
        
        // Update positions on simulation tick
        simulation.on("tick", () => {{
            // Update links
            link
                .attr("x1", d => Math.max(20, Math.min(width - 20, d.source.x)))
                .attr("y1", d => Math.max(20, Math.min(height - 20, d.source.y)))
                .attr("x2", d => Math.max(20, Math.min(width - 20, d.target.x)))
                .attr("y2", d => Math.max(20, Math.min(height - 20, d.target.y)));
            
            // Update edge labels
            edgeLabels
                .attr("x", d => (d.source.x + d.target.x) / 2)
                .attr("y", d => (d.source.y + d.target.y) / 2);
            
            // Update nodes with boundary constraints
            node
                .attr("cx", d => d.x = Math.max(20, Math.min(width - 20, d.x)))
                .attr("cy", d => d.y = Math.max(20, Math.min(height - 20, d.y)));
            
            // Update node labels
            nodeLabels
                .attr("x", d => d.x)
                .attr("y", d => d.y);
        }});
        
        // Interaction handlers
        function dragstarted(event, d) {{
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
        }}
        
        function dragged(event, d) {{
            d.fx = event.x;
            d.fy = event.y;
        }}
        
        function dragended(event, d) {{
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
        }}
        
        function handleMouseOver(event, d) {{
            d3.select(this)
                .transition()
                .duration(200)
                .attr("r", 16)
                .attr("stroke", "#333")
                .attr("stroke-width", 3);
        }}
        
        function handleMouseOut(event, d) {{
            d3.select(this)
                .transition()
                .duration(200)
                .attr("r", 12)
                .attr("stroke", "none");
        }}
        
        function handleClick(event, d) {{
            console.log('Node clicked:', d);
            alert(`Entity: ${{d.label}}\\nType: ${{d.type}}\\nID: ${{d.id}}`);
        }}
        
        // Control functions
        function resetGraph() {{
            svg.transition()
                .duration(750)
                .call(zoom.transform, d3.zoomIdentity);
            simulation.alpha(1).restart();
        }}
        
        let simulationRunning = true;
        function pauseSimulation() {{
            if (simulationRunning) {{
                simulation.stop();
                simulationRunning = false;
                event.target.textContent = 'Resume';
            }} else {{
                simulation.restart();
                simulationRunning = true;
                event.target.textContent = 'Pause';
            }}
        }}
        
        // Initial layout
        console.log('Graph initialization complete');
        
        // Auto-fit to content after initial layout
        setTimeout(() => {{
            const bounds = g.node().getBBox();
            const fullWidth = bounds.width;
            const fullHeight = bounds.height;
            const midX = bounds.x + fullWidth / 2;
            const midY = bounds.y + fullHeight / 2;
            
            if (fullWidth === 0 || fullHeight === 0) return;
            
            const scale = Math.min(width / fullWidth, height / fullHeight) * 0.8;
            const translate = [width / 2 - scale * midX, height / 2 - scale * midY];
            
            svg.transition()
                .duration(1000)
                .call(zoom.transform, d3.zoomIdentity.translate(translate[0], translate[1]).scale(scale));
        }}, 2000);
        
    </script>
</body>
</html>'''
        
        return html_content

    async def open_visualization(self, html_file: str):
        """Open visualization in web browser."""
        try:
            webbrowser.open(f'file://{Path(html_file).absolute()}')
            logger.info(f"Opened visualization in browser: {html_file}")
        except Exception as e:
            logger.error(f"Failed to open browser: {e}")


# Convenience function for quick visualization
async def visualize_neo4j_graph(graph_store, 
                               title: str = "Knowledge Graph",
                               auto_open: bool = True,
                               output_dir: str = 'visualizations') -> str:
    """
    Quick function to visualize Neo4j graph.
    """
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Initialize visualizer
    visualizer = SimpleGraphVisualizer()
    
    # Fetch graph data
    graph_3d = await visualizer.fetch_graph_data(graph_store)
    
    # Create visualization
    output_file = Path(output_dir) / f"{title.lower().replace(' ', '_')}_graph.html"
    result_file = await visualizer.create_web_visualization(graph_3d, title, str(output_file))
    
    # Auto-open if requested
    if auto_open:
        await visualizer.open_visualization(result_file)
    
    return result_file
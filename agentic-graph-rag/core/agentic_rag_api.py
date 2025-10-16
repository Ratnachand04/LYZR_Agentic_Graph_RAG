#!/usr/bin/env python3
"""
Agentic RAG API - REST API for Agentic Graph RAG System
Streaming responses, Web/REST APIs, and real-time query processing.
"""

import logging
import asyncio
import json
from typing import List, Dict, Any, Optional, AsyncGenerator
from datetime import datetime
import uuid
from pathlib import Path

try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
    from fastapi.responses import StreamingResponse, JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    logging.warning("FastAPI not available. Install with: pip install fastapi uvicorn websockets")

from .query_analyzer import QueryAnalyzer, QueryPlan, analyze_user_query
from .reasoning_engine_unified import ReasoningEngine, ReasoningResult, ReasoningContext
from .faiss_index_manager import FAISSIndexManager
from .cypher_traversal_engine import CypherTraversalEngine

logger = logging.getLogger(__name__)


# Pydantic models for API
class QueryRequest(BaseModel):
    """Request model for query endpoint."""
    query: str = Field(..., description="User query string")
    context: Optional[Dict[str, Any]] = Field(None, description="Optional context from previous queries")
    stream: bool = Field(False, description="Whether to stream the response")
    include_reasoning: bool = Field(True, description="Include detailed reasoning steps")
    confidence_threshold: float = Field(0.7, description="Minimum confidence threshold")


class QueryResponse(BaseModel):
    """Response model for query endpoint."""
    query_id: str
    query: str
    answer: str
    confidence: float
    reasoning_steps: Optional[List[Dict[str, Any]]] = None
    evidence: Optional[List[Dict[str, Any]]] = None
    alternatives: Optional[List[str]] = None
    uncertainties: Optional[List[str]] = None
    execution_time: float
    timestamp: str


class SystemStatus(BaseModel):
    """System status model."""
    status: str
    components: Dict[str, bool]
    statistics: Dict[str, Any]
    timestamp: str


class GraphUpdateRequest(BaseModel):
    """Request to update graph data."""
    entities: Optional[List[Dict[str, Any]]] = None
    relationships: Optional[List[Dict[str, Any]]] = None
    embeddings: Optional[List[Dict[str, Any]]] = None
    operation: str = Field("add", description="Operation: add, update, delete")


class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.query_subscribers: Dict[str, List[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        
        # Remove from query subscriptions
        for query_id, subscribers in self.query_subscribers.items():
            if websocket in subscribers:
                subscribers.remove(websocket)
        
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def subscribe_to_query(self, websocket: WebSocket, query_id: str):
        """Subscribe websocket to query updates."""
        if query_id not in self.query_subscribers:
            self.query_subscribers[query_id] = []
        self.query_subscribers[query_id].append(websocket)
    
    async def broadcast_to_query(self, query_id: str, message: Dict[str, Any]):
        """Broadcast message to subscribers of a specific query."""
        if query_id in self.query_subscribers:
            disconnected = []
            for websocket in self.query_subscribers[query_id]:
                try:
                    await websocket.send_json(message)
                except:
                    disconnected.append(websocket)
            
            # Remove disconnected websockets
            for ws in disconnected:
                if ws in self.query_subscribers[query_id]:
                    self.query_subscribers[query_id].remove(ws)
    
    async def broadcast_system_update(self, message: Dict[str, Any]):
        """Broadcast system update to all connected clients."""
        disconnected = []
        for websocket in self.active_connections:
            try:
                await websocket.send_json(message)
            except:
                disconnected.append(websocket)
        
        # Remove disconnected websockets
        for ws in disconnected:
            self.disconnect(ws)


class AgenticRAGAPI:
    """
    Main Agentic RAG API server.
    
    Features:
    - RESTful API endpoints for query processing
    - WebSocket support for real-time streaming
    - Graph data management and updates
    - System monitoring and statistics
    - Background task processing
    - CORS support for web integration
    """
    
    def __init__(self, 
                 vector_index: FAISSIndexManager,
                 graph_engine: CypherTraversalEngine,
                 reasoning_engine: ReasoningEngine):
        """Initialize Agentic RAG API."""
        if not FASTAPI_AVAILABLE:
            raise ImportError("FastAPI is required. Install with: pip install fastapi uvicorn websockets")
        
        self.vector_index = vector_index
        self.graph_engine = graph_engine
        self.reasoning_engine = reasoning_engine
        self.query_analyzer = QueryAnalyzer()
        
        # Connection management
        self.connection_manager = ConnectionManager()
        
        # Query tracking
        self.active_queries: Dict[str, Dict[str, Any]] = {}
        self.query_history: List[Dict[str, Any]] = []
        
        # Create FastAPI app
        self.app = FastAPI(
            title="Agentic Graph RAG API",
            description="Advanced knowledge graph reasoning with streaming responses",
            version="1.0.0"
        )
        
        # Configure CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Register routes
        self._register_routes()
        
        logger.info("Agentic RAG API initialized")
    
    def _register_routes(self):
        """Register API routes."""
        
        @self.app.get("/", response_model=Dict[str, str])
        async def root():
            """API root endpoint."""
            return {
                "message": "Agentic Graph RAG API",
                "version": "1.0.0",
                "status": "active"
            }
        
        @self.app.get("/health", response_model=SystemStatus)
        async def health_check():
            """Health check endpoint."""
            return await self._get_system_status()
        
        @self.app.post("/query", response_model=QueryResponse)
        async def process_query(request: QueryRequest, background_tasks: BackgroundTasks):
            """Process a user query with optional streaming."""
            if request.stream:
                # For streaming, return StreamingResponse
                return StreamingResponse(
                    self._stream_query_response(request),
                    media_type="application/json"
                )
            else:
                # Regular synchronous response
                return await self._process_query_sync(request, background_tasks)
        
        @self.app.post("/query/stream")
        async def stream_query(request: QueryRequest):
            """Stream query processing in real-time."""
            return StreamingResponse(
                self._stream_query_response(request),
                media_type="text/plain"
            )
        
        @self.app.get("/query/{query_id}")
        async def get_query_status(query_id: str):
            """Get status of a specific query."""
            if query_id in self.active_queries:
                return self.active_queries[query_id]
            elif any(q.get('query_id') == query_id for q in self.query_history):
                # Find in history
                query_data = next(q for q in self.query_history if q.get('query_id') == query_id)
                return query_data
            else:
                raise HTTPException(status_code=404, detail="Query not found")
        
        @self.app.get("/queries/active")
        async def get_active_queries():
            """Get all active queries."""
            return {
                "active_queries": list(self.active_queries.keys()),
                "count": len(self.active_queries)
            }
        
        @self.app.get("/queries/history")
        async def get_query_history(limit: int = 50):
            """Get query history."""
            return {
                "queries": self.query_history[-limit:],
                "count": len(self.query_history)
            }
        
        @self.app.post("/graph/update")
        async def update_graph(request: GraphUpdateRequest, background_tasks: BackgroundTasks):
            """Update graph data."""
            background_tasks.add_task(self._update_graph_data, request)
            return {"message": "Graph update initiated", "status": "processing"}
        
        @self.app.get("/graph/statistics")
        async def get_graph_statistics():
            """Get graph statistics."""
            return await self._get_graph_statistics()
        
        @self.app.get("/vector/statistics") 
        async def get_vector_statistics():
            """Get vector index statistics."""
            return await self.vector_index.get_statistics()
        
        @self.app.websocket("/ws/{client_id}")
        async def websocket_endpoint(websocket: WebSocket, client_id: str):
            """WebSocket endpoint for real-time updates."""
            await self.connection_manager.connect(websocket)
            try:
                while True:
                    # Wait for client messages
                    data = await websocket.receive_json()
                    await self._handle_websocket_message(websocket, client_id, data)
            except WebSocketDisconnect:
                self.connection_manager.disconnect(websocket)
        
        @self.app.websocket("/ws/query/{query_id}")
        async def query_websocket(websocket: WebSocket, query_id: str):
            """WebSocket endpoint for specific query updates."""
            await self.connection_manager.connect(websocket)
            await self.connection_manager.subscribe_to_query(websocket, query_id)
            try:
                while True:
                    # Keep connection alive and wait for disconnection
                    await asyncio.sleep(1)
            except WebSocketDisconnect:
                self.connection_manager.disconnect(websocket)
    
    async def _process_query_sync(self, request: QueryRequest, background_tasks: BackgroundTasks) -> QueryResponse:
        """Process query synchronously."""
        query_id = str(uuid.uuid4())
        start_time = datetime.now()
        
        try:
            # Track active query
            self.active_queries[query_id] = {
                "query_id": query_id,
                "query": request.query,
                "status": "processing",
                "started_at": start_time.isoformat()
            }
            
            # Analyze query
            query_plan = await analyze_user_query(request.query, request.context)
            
            # Apply confidence threshold
            query_plan.confidence_threshold = request.confidence_threshold
            
            # Execute reasoning
            reasoning_context = ReasoningContext(
                query_id=query_id,
                confidence_threshold=request.confidence_threshold
            )
            
            reasoning_result = await self.reasoning_engine.reason(query_plan, reasoning_context)
            
            # Prepare response
            execution_time = (datetime.now() - start_time).total_seconds()
            
            response = QueryResponse(
                query_id=query_id,
                query=request.query,
                answer=reasoning_result.answer,
                confidence=reasoning_result.confidence,
                reasoning_steps=[step.to_dict() for step in reasoning_result.reasoning_steps] if request.include_reasoning else None,
                evidence=[ev.to_dict() for ev in reasoning_result.evidence_used] if request.include_reasoning else None,
                alternatives=reasoning_result.alternative_explanations,
                uncertainties=reasoning_result.uncertainty_factors,
                execution_time=execution_time,
                timestamp=datetime.now().isoformat()
            )
            
            # Update tracking
            self.active_queries[query_id]["status"] = "completed"
            self.active_queries[query_id]["result"] = response.dict()
            
            # Move to history
            background_tasks.add_task(self._archive_query, query_id)
            
            return response
        
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            
            # Update tracking with error
            self.active_queries[query_id]["status"] = "error"
            self.active_queries[query_id]["error"] = str(e)
            
            raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")
    
    async def _stream_query_response(self, request: QueryRequest) -> AsyncGenerator[str, None]:
        """Stream query processing results."""
        query_id = str(uuid.uuid4())
        start_time = datetime.now()
        
        try:
            # Send initial response
            yield json.dumps({
                "type": "query_start",
                "query_id": query_id,
                "query": request.query,
                "timestamp": start_time.isoformat()
            }) + "\n"
            
            # Track active query
            self.active_queries[query_id] = {
                "query_id": query_id,
                "query": request.query,
                "status": "streaming",
                "started_at": start_time.isoformat()
            }
            
            # Analyze query
            yield json.dumps({
                "type": "analysis_start",
                "message": "Analyzing query..."
            }) + "\n"
            
            query_plan = await analyze_user_query(request.query, request.context)
            
            yield json.dumps({
                "type": "analysis_complete",
                "intent": query_plan.intent.to_dict(),
                "complexity": query_plan.decomposition.complexity.value,
                "strategies": query_plan.search_strategies
            }) + "\n"
            
            # Stream reasoning
            reasoning_context = ReasoningContext(
                query_id=query_id,
                confidence_threshold=request.confidence_threshold
            )
            
            reasoning_stream = await self.reasoning_engine.reason(
                query_plan, reasoning_context, stream_results=True
            )
            
            # Stream reasoning updates
            async for reasoning_update in reasoning_stream:
                yield json.dumps(reasoning_update) + "\n"
                
                # Also broadcast via WebSocket
                await self.connection_manager.broadcast_to_query(query_id, reasoning_update)
            
            # Mark as completed
            self.active_queries[query_id]["status"] = "completed"
            
            yield json.dumps({
                "type": "query_complete",
                "query_id": query_id,
                "execution_time": (datetime.now() - start_time).total_seconds()
            }) + "\n"
        
        except Exception as e:
            logger.error(f"Streaming query failed: {e}")
            
            yield json.dumps({
                "type": "error",
                "query_id": query_id,
                "error": str(e)
            }) + "\n"
            
            self.active_queries[query_id]["status"] = "error"
            self.active_queries[query_id]["error"] = str(e)
    
    async def _handle_websocket_message(self, websocket: WebSocket, client_id: str, data: Dict[str, Any]):
        """Handle incoming WebSocket messages."""
        message_type = data.get("type")
        
        if message_type == "subscribe_query":
            query_id = data.get("query_id")
            if query_id:
                await self.connection_manager.subscribe_to_query(websocket, query_id)
                await websocket.send_json({
                    "type": "subscription_confirmed",
                    "query_id": query_id
                })
        
        elif message_type == "ping":
            await websocket.send_json({"type": "pong"})
        
        elif message_type == "get_status":
            status = await self._get_system_status()
            await websocket.send_json({
                "type": "system_status",
                "data": status.dict()
            })
    
    async def _archive_query(self, query_id: str):
        """Move completed query to history."""
        if query_id in self.active_queries:
            query_data = self.active_queries.pop(query_id)
            self.query_history.append(query_data)
            
            # Keep history limited
            if len(self.query_history) > 1000:
                self.query_history = self.query_history[-500:]
    
    async def _update_graph_data(self, request: GraphUpdateRequest):
        """Update graph data in background."""
        try:
            # Update vector index if embeddings provided
            if request.embeddings and request.operation in ["add", "update"]:
                embeddings = [item["embedding"] for item in request.embeddings]
                ids = [item["id"] for item in request.embeddings]
                metadata = [item.get("metadata", {}) for item in request.embeddings]
                
                await self.vector_index.add_embeddings(embeddings, ids, metadata)
            
            # Update graph database if entities/relationships provided
            if request.entities or request.relationships:
                # This would integrate with the graph engine
                # Implementation depends on specific graph update requirements
                pass
            
            # Broadcast update notification
            await self.connection_manager.broadcast_system_update({
                "type": "graph_updated",
                "operation": request.operation,
                "timestamp": datetime.now().isoformat()
            })
            
            logger.info(f"Graph data updated: {request.operation}")
        
        except Exception as e:
            logger.error(f"Graph update failed: {e}")
    
    async def _get_system_status(self) -> SystemStatus:
        """Get current system status."""
        # Check component health
        components = {
            "vector_index": self.vector_index is not None,
            "graph_engine": self.graph_engine.connected if self.graph_engine else False,
            "reasoning_engine": self.reasoning_engine is not None,
            "query_analyzer": self.query_analyzer is not None
        }
        
        # Get statistics
        statistics = {
            "active_queries": len(self.active_queries),
            "total_queries": len(self.query_history) + len(self.active_queries),
            "active_connections": len(self.connection_manager.active_connections),
            "uptime": "N/A"  # Would need startup time tracking
        }
        
        # Add vector statistics if available
        try:
            vector_stats = await self.vector_index.get_statistics()
            statistics["vector_index"] = vector_stats
        except:
            pass
        
        # Determine overall status
        overall_status = "healthy" if all(components.values()) else "degraded"
        
        return SystemStatus(
            status=overall_status,
            components=components,
            statistics=statistics,
            timestamp=datetime.now().isoformat()
        )
    
    async def _get_graph_statistics(self) -> Dict[str, Any]:
        """Get graph database statistics."""
        try:
            if self.graph_engine and self.graph_engine.connected:
                return await self.graph_engine.analyze_graph_structure()
            else:
                return {"error": "Graph engine not connected"}
        except Exception as e:
            logger.error(f"Failed to get graph statistics: {e}")
            return {"error": str(e)}
    
    def run(self, host: str = "0.0.0.0", port: int = 8000, **kwargs):
        """Run the API server."""
        logger.info(f"Starting Agentic RAG API server on {host}:{port}")
        uvicorn.run(self.app, host=host, port=port, **kwargs)


# Convenience functions
async def create_agentic_rag_api(vector_index: FAISSIndexManager,
                               graph_engine: CypherTraversalEngine,
                               reasoning_engine: ReasoningEngine) -> AgenticRAGAPI:
    """Create Agentic RAG API server."""
    return AgenticRAGAPI(vector_index, graph_engine, reasoning_engine)


def run_agentic_rag_server(vector_index: FAISSIndexManager,
                          graph_engine: CypherTraversalEngine, 
                          reasoning_engine: ReasoningEngine,
                          host: str = "0.0.0.0",
                          port: int = 8000):
    """Run Agentic RAG API server."""
    api = AgenticRAGAPI(vector_index, graph_engine, reasoning_engine)
    api.run(host=host, port=port)


# Example usage and testing
if __name__ == "__main__":
    # This would be used for standalone server execution
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    async def main():
        # Initialize components (would need actual implementations)
        # vector_index = await create_index_manager()
        # graph_engine = await create_cypher_engine(...)
        # reasoning_engine = create_reasoning_engine(vector_index, graph_engine)
        
        # api = await create_agentic_rag_api(vector_index, graph_engine, reasoning_engine)
        # api.run()
        
        print("Agentic RAG API - Example implementation")
        print("Install required dependencies: pip install fastapi uvicorn websockets")
    
    asyncio.run(main())

# Create a module-level app instance for uvicorn
# This will be initialized by the launcher
app = None
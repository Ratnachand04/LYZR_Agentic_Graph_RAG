#!/usr/bin/env python3
"""
Reasoning Engine for Agentic Graph RAG
Advanced multi-step reasoning system with context management and inference chains.
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple, Set, Union, AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json
import uuid

from .query_analyzer import QueryPlan, QueryType, QueryComplexity
from .faiss_index_manager import FAISSIndexManager, VectorSearchResult
from .cypher_traversal_engine import CypherTraversalEngine, TraversalResult

logger = logging.getLogger(__name__)


class ReasoningType(Enum):
    """Types of reasoning strategies."""
    DEDUCTIVE = "deductive"        # General to specific
    INDUCTIVE = "inductive"        # Specific to general  
    ABDUCTIVE = "abductive"        # Best explanation
    ANALOGICAL = "analogical"      # Similarity-based
    CAUSAL = "causal"             # Cause-effect chains
    TEMPORAL = "temporal"          # Time-based reasoning
    SPATIAL = "spatial"           # Location-based reasoning
    COMPOSITIONAL = "compositional" # Part-whole reasoning
    STATISTICAL = "statistical"   # Statistical/numerical reasoning


class EvidenceType(Enum):
    """Types of evidence for reasoning."""
    DIRECT_FACT = "direct_fact"
    INFERRED_FACT = "inferred_fact"
    STATISTICAL = "statistical"
    ANALOGICAL = "analogical"
    EXPERT_OPINION = "expert_opinion"
    CONTEXTUAL = "contextual"


@dataclass
class Evidence:
    """Represents a piece of evidence for reasoning."""
    content: str
    source: str
    evidence_type: EvidenceType
    confidence: float
    relevance: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'content': self.content,
            'source': self.source,
            'evidence_type': self.evidence_type.value,
            'confidence': self.confidence,
            'relevance': self.relevance,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }


@dataclass
class ReasoningStep:
    """Represents a single step in reasoning chain."""
    step_id: str
    description: str
    reasoning_type: ReasoningType
    input_evidence: List[Evidence]
    output_conclusion: str
    confidence: float
    reasoning_chain: str
    dependencies: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'step_id': self.step_id,
            'description': self.description,
            'reasoning_type': self.reasoning_type.value,
            'input_evidence': [e.to_dict() for e in self.input_evidence],
            'output_conclusion': self.output_conclusion,
            'confidence': self.confidence,
            'reasoning_chain': self.reasoning_chain,
            'dependencies': self.dependencies
        }


@dataclass
class ReasoningContext:
    """Maintains context throughout reasoning process."""
    query_id: str
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    known_facts: Dict[str, Evidence] = field(default_factory=dict)
    working_memory: Dict[str, Any] = field(default_factory=dict)
    reasoning_trace: List[ReasoningStep] = field(default_factory=list)
    confidence_threshold: float = 0.7
    
    def add_fact(self, fact_id: str, evidence: Evidence):
        """Add a fact to working memory."""
        self.known_facts[fact_id] = evidence
    
    def get_relevant_facts(self, keywords: List[str]) -> List[Evidence]:
        """Get facts relevant to keywords."""
        relevant = []
        for evidence in self.known_facts.values():
            for keyword in keywords:
                if keyword.lower() in evidence.content.lower():
                    relevant.append(evidence)
                    break
        return relevant
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'query_id': self.query_id,
            'conversation_history': self.conversation_history,
            'known_facts': {k: v.to_dict() for k, v in self.known_facts.items()},
            'working_memory': self.working_memory,
            'reasoning_trace': [step.to_dict() for step in self.reasoning_trace],
            'confidence_threshold': self.confidence_threshold
        }


@dataclass
class ReasoningResult:
    """Final result from reasoning process."""
    query: str
    answer: str
    confidence: float
    reasoning_steps: List[ReasoningStep]
    evidence_used: List[Evidence]
    alternative_explanations: List[str]
    uncertainty_factors: List[str]
    reasoning_time: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'query': self.query,
            'answer': self.answer,
            'confidence': self.confidence,
            'reasoning_steps': [step.to_dict() for step in self.reasoning_steps],
            'evidence_used': [e.to_dict() for e in self.evidence_used],
            'alternative_explanations': self.alternative_explanations,
            'uncertainty_factors': self.uncertainty_factors,
            'reasoning_time': self.reasoning_time
        }


class ReasoningEngine:
    """
    Advanced reasoning engine for knowledge graph analysis.
    
    Features:
    - Multi-step reasoning chains
    - Evidence collection and evaluation
    - Confidence tracking and uncertainty management
    - Context-aware inference
    - Streaming reasoning for real-time updates
    - Alternative hypothesis generation
    - Explanation generation
    """
    
    def __init__(self, 
                 vector_index: FAISSIndexManager,
                 graph_engine: CypherTraversalEngine,
                 llm_client: Optional[Any] = None):
        """Initialize reasoning engine."""
        self.vector_index = vector_index
        self.graph_engine = graph_engine
        self.llm_client = llm_client
        
        # Reasoning strategies
        self.reasoning_strategies = {
            QueryType.FACTUAL: [ReasoningType.DEDUCTIVE],
            QueryType.CONCEPTUAL: [ReasoningType.DEDUCTIVE, ReasoningType.COMPOSITIONAL],
            QueryType.RELATIONAL: [ReasoningType.ANALOGICAL, ReasoningType.DEDUCTIVE],
            QueryType.ANALYTICAL: [ReasoningType.INDUCTIVE, ReasoningType.STATISTICAL],
            QueryType.CAUSAL: [ReasoningType.CAUSAL, ReasoningType.ABDUCTIVE],
            QueryType.COMPARATIVE: [ReasoningType.ANALOGICAL, ReasoningType.DEDUCTIVE],
            QueryType.TEMPORAL: [ReasoningType.TEMPORAL, ReasoningType.CAUSAL],
            QueryType.PROCEDURAL: [ReasoningType.DEDUCTIVE, ReasoningType.COMPOSITIONAL]
        }
        
        # Confidence thresholds by reasoning type
        self.reasoning_thresholds = {
            ReasoningType.DEDUCTIVE: 0.8,
            ReasoningType.INDUCTIVE: 0.6,
            ReasoningType.ABDUCTIVE: 0.5,
            ReasoningType.ANALOGICAL: 0.6,
            ReasoningType.CAUSAL: 0.7,
            ReasoningType.TEMPORAL: 0.7,
            ReasoningType.SPATIAL: 0.7,
            ReasoningType.COMPOSITIONAL: 0.8
        }
        
        logger.info("Reasoning Engine initialized")
    
    async def reason(self, 
                    query_plan: QueryPlan,
                    context: Optional[ReasoningContext] = None,
                    stream_results: bool = False) -> Union[ReasoningResult, AsyncGenerator[Dict[str, Any], None]]:
        """
        Execute reasoning process for a query.
        
        Args:
            query_plan: Analyzed query plan
            context: Optional reasoning context
            stream_results: Whether to stream intermediate results
            
        Returns:
            ReasoningResult or async generator for streaming
        """
        start_time = datetime.now()
        
        # Initialize context if not provided
        if context is None:
            context = ReasoningContext(
                query_id=str(uuid.uuid4()),
                confidence_threshold=query_plan.confidence_threshold
            )
        
        logger.info(f"Starting reasoning for query: {query_plan.query}")
        
        if stream_results:
            return self._stream_reasoning(query_plan, context, start_time)
        else:
            return await self._execute_reasoning(query_plan, context, start_time)
    
    async def _execute_reasoning(self, 
                               query_plan: QueryPlan, 
                               context: ReasoningContext,
                               start_time: datetime) -> ReasoningResult:
        """Execute complete reasoning process."""
        reasoning_steps = []
        all_evidence = []
        
        # Step 1: Collect initial evidence
        evidence_step = await self._collect_evidence(query_plan, context)
        reasoning_steps.append(evidence_step)
        all_evidence.extend(evidence_step.input_evidence)
        
        # Step 2: Apply reasoning strategies
        for strategy in self._select_reasoning_strategies(query_plan):
            strategy_step = await self._apply_reasoning_strategy(
                strategy, query_plan, context, all_evidence
            )
            reasoning_steps.append(strategy_step)
            
            # Update evidence with new conclusions
            conclusion_evidence = Evidence(
                content=strategy_step.output_conclusion,
                source="reasoning_engine",
                evidence_type=EvidenceType.INFERRED_FACT,
                confidence=strategy_step.confidence,
                relevance=1.0
            )
            all_evidence.append(conclusion_evidence)
        
        # Step 3: Synthesize final answer
        synthesis_step = await self._synthesize_answer(query_plan, reasoning_steps, all_evidence)
        reasoning_steps.append(synthesis_step)
        
        # Step 4: Generate alternatives and uncertainty analysis
        alternatives = await self._generate_alternatives(query_plan, reasoning_steps)
        uncertainties = await self._analyze_uncertainties(reasoning_steps, all_evidence)
        
        # Calculate overall confidence
        overall_confidence = self._calculate_overall_confidence(reasoning_steps)
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        result = ReasoningResult(
            query=query_plan.query,
            answer=synthesis_step.output_conclusion,
            confidence=overall_confidence,
            reasoning_steps=reasoning_steps,
            evidence_used=all_evidence,
            alternative_explanations=alternatives,
            uncertainty_factors=uncertainties,
            reasoning_time=execution_time
        )
        
        logger.info(f"Reasoning completed in {execution_time:.2f}s with confidence {overall_confidence:.2f}")
        return result
    
    async def _stream_reasoning(self, 
                              query_plan: QueryPlan, 
                              context: ReasoningContext,
                              start_time: datetime) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream reasoning process in real-time."""
        yield {
            'type': 'reasoning_start',
            'query': query_plan.query,
            'timestamp': datetime.now().isoformat()
        }
        
        reasoning_steps = []
        all_evidence = []
        
        # Stream evidence collection
        yield {
            'type': 'step_start',
            'step': 'evidence_collection',
            'description': 'Collecting relevant evidence...'
        }
        
        evidence_step = await self._collect_evidence(query_plan, context)
        reasoning_steps.append(evidence_step)
        all_evidence.extend(evidence_step.input_evidence)
        
        yield {
            'type': 'step_complete',
            'step': 'evidence_collection',
            'evidence_count': len(all_evidence),
            'step_data': evidence_step.to_dict()
        }
        
        # Stream reasoning strategies
        strategies = self._select_reasoning_strategies(query_plan)
        for i, strategy in enumerate(strategies):
            yield {
                'type': 'step_start',
                'step': f'reasoning_strategy_{i}',
                'strategy': strategy.value,
                'description': f'Applying {strategy.value} reasoning...'
            }
            
            strategy_step = await self._apply_reasoning_strategy(
                strategy, query_plan, context, all_evidence
            )
            reasoning_steps.append(strategy_step)
            
            yield {
                'type': 'step_complete',
                'step': f'reasoning_strategy_{i}',
                'conclusion': strategy_step.output_conclusion,
                'confidence': strategy_step.confidence,
                'step_data': strategy_step.to_dict()
            }
            
            # Update evidence
            conclusion_evidence = Evidence(
                content=strategy_step.output_conclusion,
                source="reasoning_engine",
                evidence_type=EvidenceType.INFERRED_FACT,
                confidence=strategy_step.confidence,
                relevance=1.0
            )
            all_evidence.append(conclusion_evidence)
        
        # Stream synthesis
        yield {
            'type': 'step_start',
            'step': 'synthesis',
            'description': 'Synthesizing final answer...'
        }
        
        synthesis_step = await self._synthesize_answer(query_plan, reasoning_steps, all_evidence)
        reasoning_steps.append(synthesis_step)
        
        # Calculate final confidence
        overall_confidence = self._calculate_overall_confidence(reasoning_steps)
        execution_time = (datetime.now() - start_time).total_seconds()
        
        yield {
            'type': 'reasoning_complete',
            'answer': synthesis_step.output_conclusion,
            'confidence': overall_confidence,
            'execution_time': execution_time,
            'step_count': len(reasoning_steps)
        }
    
    async def _collect_evidence(self, query_plan: QueryPlan, context: ReasoningContext) -> ReasoningStep:
        """Collect evidence from various sources."""
        step_id = f"evidence_{uuid.uuid4().hex[:8]}"
        evidence_list = []
        
        # Collect from vector similarity search
        if 'vector_similarity' in query_plan.search_strategies:
            vector_evidence = await self._collect_vector_evidence(query_plan)
            evidence_list.extend(vector_evidence)
        
        # Collect from graph traversal
        if 'graph_traversal' in query_plan.search_strategies:
            graph_evidence = await self._collect_graph_evidence(query_plan)
            evidence_list.extend(graph_evidence)
        
        # Collect from entity context
        if 'entity_context_expansion' in query_plan.search_strategies:
            entity_evidence = await self._collect_entity_evidence(query_plan)
            evidence_list.extend(entity_evidence)
        
        # Add evidence to context
        for i, evidence in enumerate(evidence_list):
            context.add_fact(f"fact_{step_id}_{i}", evidence)
        
        return ReasoningStep(
            step_id=step_id,
            description="Collect relevant evidence from knowledge sources",
            reasoning_type=ReasoningType.DEDUCTIVE,
            input_evidence=[],
            output_conclusion=f"Collected {len(evidence_list)} pieces of evidence",
            confidence=0.9,
            reasoning_chain="Evidence collection from multiple sources"
        )
    
    async def _collect_vector_evidence(self, query_plan: QueryPlan) -> List[Evidence]:
        """Collect evidence using vector similarity search."""
        evidence_list = []
        
        try:
            # Use query keywords for embedding search
            query_text = ' '.join(query_plan.intent.keywords)
            
            # This would need integration with embedding generation
            # For now, we'll use a placeholder
            results = []  # await self.vector_index.search_similar(query_embedding, k=10)
            
            for result in results:
                evidence = Evidence(
                    content=result.content,
                    source=f"vector_search_{result.id}",
                    evidence_type=EvidenceType.DIRECT_FACT,
                    confidence=result.score,
                    relevance=result.score,
                    metadata=result.metadata
                )
                evidence_list.append(evidence)
        
        except Exception as e:
            logger.warning(f"Vector evidence collection failed: {e}")
        
        return evidence_list
    
    async def _collect_graph_evidence(self, query_plan: QueryPlan) -> List[Evidence]:
        """Collect evidence using graph traversal."""
        evidence_list = []
        
        try:
            # Use entities from query for graph search
            for entity in query_plan.intent.entities:
                traversal_result = await self.graph_engine.find_entity_neighbors(entity)
                
                for node in traversal_result.nodes:
                    content = f"Entity {entity} has properties: {json.dumps(node, default=str)}"
                    evidence = Evidence(
                        content=content,
                        source=f"graph_traversal_{entity}",
                        evidence_type=EvidenceType.DIRECT_FACT,
                        confidence=0.8,
                        relevance=0.7,
                        metadata={'entity': entity, 'node_data': node}
                    )
                    evidence_list.append(evidence)
                
                for rel in traversal_result.relationships:
                    content = f"Relationship: {json.dumps(rel, default=str)}"
                    evidence = Evidence(
                        content=content,
                        source=f"graph_relationships_{entity}",
                        evidence_type=EvidenceType.DIRECT_FACT,
                        confidence=0.8,
                        relevance=0.8,
                        metadata={'entity': entity, 'relationship_data': rel}
                    )
                    evidence_list.append(evidence)
        
        except Exception as e:
            logger.warning(f"Graph evidence collection failed: {e}")
        
        return evidence_list
    
    async def _collect_entity_evidence(self, query_plan: QueryPlan) -> List[Evidence]:
        """Collect evidence about specific entities."""
        evidence_list = []
        
        for entity in query_plan.intent.entities:
            # Create evidence about entity existence and properties
            evidence = Evidence(
                content=f"Entity '{entity}' is mentioned in the query context",
                source="query_analysis",
                evidence_type=EvidenceType.CONTEXTUAL,
                confidence=0.9,
                relevance=1.0,
                metadata={'entity': entity, 'source': 'query'}
            )
            evidence_list.append(evidence)
        
        return evidence_list
    
    def _select_reasoning_strategies(self, query_plan: QueryPlan) -> List[ReasoningType]:
        """Select appropriate reasoning strategies for the query."""
        primary_strategies = self.reasoning_strategies.get(
            query_plan.intent.primary_intent, 
            [ReasoningType.DEDUCTIVE]
        )
        
        # Add strategies based on complexity
        if query_plan.decomposition.complexity in [QueryComplexity.COMPLEX, QueryComplexity.EXPERT]:
            if ReasoningType.INDUCTIVE not in primary_strategies:
                primary_strategies.append(ReasoningType.INDUCTIVE)
        
        # Add causal reasoning for "why" questions
        if any(word in query_plan.query.lower() for word in ['why', 'because', 'cause']):
            if ReasoningType.CAUSAL not in primary_strategies:
                primary_strategies.append(ReasoningType.CAUSAL)
        
        return primary_strategies
    
    async def _apply_reasoning_strategy(self, 
                                      strategy: ReasoningType,
                                      query_plan: QueryPlan,
                                      context: ReasoningContext,
                                      evidence: List[Evidence]) -> ReasoningStep:
        """Apply specific reasoning strategy."""
        step_id = f"{strategy.value}_{uuid.uuid4().hex[:8]}"
        
        # Filter relevant evidence for this strategy
        relevant_evidence = self._filter_relevant_evidence(evidence, strategy, query_plan)
        
        if strategy == ReasoningType.DEDUCTIVE:
            conclusion = await self._apply_deductive_reasoning(relevant_evidence, query_plan)
        elif strategy == ReasoningType.INDUCTIVE:
            conclusion = await self._apply_inductive_reasoning(relevant_evidence, query_plan)
        elif strategy == ReasoningType.ABDUCTIVE:
            conclusion = await self._apply_abductive_reasoning(relevant_evidence, query_plan)
        elif strategy == ReasoningType.ANALOGICAL:
            conclusion = await self._apply_analogical_reasoning(relevant_evidence, query_plan)
        elif strategy == ReasoningType.CAUSAL:
            conclusion = await self._apply_causal_reasoning(relevant_evidence, query_plan)
        else:
            conclusion = await self._apply_default_reasoning(relevant_evidence, query_plan)
        
        # Calculate confidence based on evidence quality and strategy
        confidence = self._calculate_strategy_confidence(strategy, relevant_evidence)
        
        return ReasoningStep(
            step_id=step_id,
            description=f"Apply {strategy.value} reasoning",
            reasoning_type=strategy,
            input_evidence=relevant_evidence,
            output_conclusion=conclusion,
            confidence=confidence,
            reasoning_chain=f"Used {len(relevant_evidence)} pieces of evidence with {strategy.value} reasoning"
        )
    
    def _filter_relevant_evidence(self, 
                                evidence: List[Evidence], 
                                strategy: ReasoningType,
                                query_plan: QueryPlan) -> List[Evidence]:
        """Filter evidence relevant to specific reasoning strategy."""
        relevant = []
        
        # Basic relevance filtering
        for ev in evidence:
            if ev.relevance >= 0.5:  # Minimum relevance threshold
                relevant.append(ev)
        
        # Strategy-specific filtering
        if strategy == ReasoningType.STATISTICAL:
            # Prefer statistical evidence
            relevant = [ev for ev in relevant if ev.evidence_type == EvidenceType.STATISTICAL]
        elif strategy == ReasoningType.CAUSAL:
            # Look for causal indicators in content
            causal_evidence = []
            for ev in relevant:
                if any(word in ev.content.lower() for word in ['cause', 'effect', 'because', 'result']):
                    causal_evidence.append(ev)
            if causal_evidence:
                relevant = causal_evidence
        
        # Sort by relevance and confidence
        relevant.sort(key=lambda x: (x.relevance + x.confidence) / 2, reverse=True)
        
        return relevant[:10]  # Limit to top 10 pieces
    
    async def _apply_deductive_reasoning(self, evidence: List[Evidence], query_plan: QueryPlan) -> str:
        """Apply deductive reasoning: general principles to specific conclusions."""
        if not evidence:
            return "Insufficient evidence for deductive reasoning"
        
        # Look for general facts that can be applied to specific query
        general_facts = [ev for ev in evidence if ev.evidence_type == EvidenceType.DIRECT_FACT]
        
        if general_facts:
            # Simple deductive conclusion
            primary_fact = general_facts[0]
            conclusion = f"Based on the established fact that {primary_fact.content}, "
            
            # Apply to specific entities in query
            if query_plan.intent.entities:
                entity = query_plan.intent.entities[0]
                conclusion += f"we can conclude that {entity} follows this pattern."
            else:
                conclusion += "this principle applies to the query context."
        else:
            conclusion = "Available evidence supports a deductive conclusion, but more specific facts are needed."
        
        return conclusion
    
    async def _apply_inductive_reasoning(self, evidence: List[Evidence], query_plan: QueryPlan) -> str:
        """Apply inductive reasoning: specific observations to general patterns."""
        if len(evidence) < 2:
            return "Insufficient evidence for inductive reasoning (need multiple observations)"
        
        # Look for patterns across multiple pieces of evidence
        patterns = {}
        for ev in evidence:
            # Simple pattern detection based on common words
            words = ev.content.lower().split()
            for word in words:
                if len(word) > 3:  # Ignore short words
                    patterns[word] = patterns.get(word, 0) + 1
        
        # Find most common patterns
        common_patterns = sorted(patterns.items(), key=lambda x: x[1], reverse=True)[:3]
        
        if common_patterns:
            pattern_words = [word for word, count in common_patterns if count > 1]
            conclusion = f"Based on multiple observations, there appears to be a pattern involving: {', '.join(pattern_words)}. "
            conclusion += f"This suggests a general principle that can be applied more broadly."
        else:
            conclusion = "Multiple pieces of evidence examined, but no clear patterns emerge for generalization."
        
        return conclusion
    
    async def _apply_abductive_reasoning(self, evidence: List[Evidence], query_plan: QueryPlan) -> str:
        """Apply abductive reasoning: find best explanation for observations."""
        if not evidence:
            return "No evidence available for abductive reasoning"
        
        # Generate possible explanations based on evidence
        explanations = []
        
        # Look at evidence types to generate hypotheses
        fact_count = sum(1 for ev in evidence if ev.evidence_type == EvidenceType.DIRECT_FACT)
        inferred_count = sum(1 for ev in evidence if ev.evidence_type == EvidenceType.INFERRED_FACT)
        
        if fact_count > inferred_count:
            explanations.append("The most likely explanation is based on direct factual evidence")
        elif inferred_count > 0:
            explanations.append("The explanation involves inferred relationships and patterns")
        
        # Consider confidence levels
        high_conf_evidence = [ev for ev in evidence if ev.confidence > 0.8]
        if high_conf_evidence:
            explanations.append(f"High-confidence evidence ({len(high_conf_evidence)} sources) supports this explanation")
        
        if explanations:
            conclusion = f"The best explanation appears to be: {explanations[0]}. "
            if len(explanations) > 1:
                conclusion += f"Supporting factors: {'; '.join(explanations[1:])}"
        else:
            conclusion = "Multiple explanations are possible, but evidence is insufficient to determine the most likely one."
        
        return conclusion
    
    async def _apply_analogical_reasoning(self, evidence: List[Evidence], query_plan: QueryPlan) -> str:
        """Apply analogical reasoning: reasoning by similarity and comparison."""
        if len(evidence) < 2:
            return "Insufficient evidence for analogical reasoning (need comparable cases)"
        
        # Look for similar patterns or structures
        similarities = []
        
        # Simple similarity detection based on shared words/concepts
        for i, ev1 in enumerate(evidence[:-1]):
            for ev2 in evidence[i+1:]:
                shared_words = set(ev1.content.lower().split()) & set(ev2.content.lower().split())
                meaningful_shared = [w for w in shared_words if len(w) > 3]
                
                if len(meaningful_shared) > 2:
                    similarities.append(f"Similar pattern found: {', '.join(meaningful_shared[:3])}")
        
        if similarities:
            conclusion = f"Analogical reasoning reveals similar patterns: {similarities[0]}. "
            conclusion += "This suggests that similar principles or mechanisms are at work."
        else:
            conclusion = "Analogical reasoning applied, but no clear similarities found between available evidence."
        
        return conclusion
    
    async def _apply_causal_reasoning(self, evidence: List[Evidence], query_plan: QueryPlan) -> str:
        """Apply causal reasoning: identify cause-effect relationships."""
        # Look for causal indicators in evidence
        causal_evidence = []
        
        for ev in evidence:
            causal_words = ['cause', 'effect', 'result', 'lead to', 'because', 'due to', 'consequence']
            if any(word in ev.content.lower() for word in causal_words):
                causal_evidence.append(ev)
        
        if causal_evidence:
            # Analyze causal chain
            conclusion = "Causal analysis reveals: "
            
            # Try to identify causes and effects
            causes = []
            effects = []
            
            for ev in causal_evidence:
                content_lower = ev.content.lower()
                if 'cause' in content_lower or 'because' in content_lower:
                    causes.append(ev.content[:100] + "...")
                if 'effect' in content_lower or 'result' in content_lower:
                    effects.append(ev.content[:100] + "...")
            
            if causes and effects:
                conclusion += f"Identified causes: {'; '.join(causes[:2])} leading to effects: {'; '.join(effects[:2])}"
            elif causes:
                conclusion += f"Identified potential causes: {'; '.join(causes[:2])}"
            elif effects:
                conclusion += f"Identified potential effects: {'; '.join(effects[:2])}"
            else:
                conclusion += "Causal relationships exist but require further analysis to clarify the chain."
        else:
            conclusion = "Causal reasoning attempted, but no clear causal relationships found in available evidence."
        
        return conclusion
    
    async def _apply_default_reasoning(self, evidence: List[Evidence], query_plan: QueryPlan) -> str:
        """Apply default reasoning when specific strategies don't apply."""
        if not evidence:
            return "No evidence available for reasoning"
        
        # Synthesize based on evidence confidence and relevance
        high_quality_evidence = [ev for ev in evidence if ev.confidence > 0.7 and ev.relevance > 0.6]
        
        if high_quality_evidence:
            conclusion = f"Based on {len(high_quality_evidence)} high-quality sources, "
            
            # Extract key points from top evidence
            top_evidence = high_quality_evidence[0]
            conclusion += f"the primary finding is: {top_evidence.content[:200]}..."
            
            if len(high_quality_evidence) > 1:
                conclusion += f" Additional evidence from {len(high_quality_evidence)-1} sources provides supporting context."
        else:
            conclusion = f"Analysis of {len(evidence)} sources provides relevant information, but confidence levels suggest further investigation may be needed."
        
        return conclusion
    
    def _calculate_strategy_confidence(self, strategy: ReasoningType, evidence: List[Evidence]) -> float:
        """Calculate confidence for a reasoning strategy based on evidence quality."""
        if not evidence:
            return 0.1
        
        # Base confidence from strategy
        base_confidence = self.reasoning_thresholds.get(strategy, 0.7)
        
        # Adjust based on evidence quality
        avg_evidence_conf = sum(ev.confidence for ev in evidence) / len(evidence)
        avg_relevance = sum(ev.relevance for ev in evidence) / len(evidence)
        
        # Weighted combination
        strategy_confidence = (
            base_confidence * 0.4 +
            avg_evidence_conf * 0.4 +
            avg_relevance * 0.2
        )
        
        # Penalty for insufficient evidence
        if len(evidence) < 2:
            strategy_confidence *= 0.8
        
        return min(max(strategy_confidence, 0.1), 0.95)  # Clamp between 0.1 and 0.95
    
    async def _synthesize_answer(self, 
                               query_plan: QueryPlan, 
                               reasoning_steps: List[ReasoningStep],
                               all_evidence: List[Evidence]) -> ReasoningStep:
        """Synthesize final answer from reasoning steps."""
        step_id = f"synthesis_{uuid.uuid4().hex[:8]}"
        
        # Collect conclusions from reasoning steps
        conclusions = [step.output_conclusion for step in reasoning_steps[1:]]  # Skip evidence collection step
        
        # Weight conclusions by confidence
        weighted_conclusions = []
        total_weight = 0
        
        for step in reasoning_steps[1:]:
            weight = step.confidence
            weighted_conclusions.append((step.output_conclusion, weight))
            total_weight += weight
        
        # Generate synthesis
        if weighted_conclusions:
            # Start with highest confidence conclusion
            best_conclusion = max(weighted_conclusions, key=lambda x: x[1])
            
            synthesis = f"Based on the reasoning analysis: {best_conclusion[0]} "
            
            # Add supporting conclusions if available
            supporting = [concl for concl, weight in weighted_conclusions 
                         if concl != best_conclusion[0] and weight > 0.6]
            
            if supporting:
                synthesis += f"This is further supported by additional analysis showing {supporting[0][:100]}..."
        else:
            synthesis = "Analysis completed, but no definitive conclusions could be drawn from available evidence."
        
        # Calculate synthesis confidence
        synthesis_confidence = total_weight / len(reasoning_steps[1:]) if reasoning_steps[1:] else 0.5
        
        return ReasoningStep(
            step_id=step_id,
            description="Synthesize final answer from reasoning chain",
            reasoning_type=ReasoningType.DEDUCTIVE,
            input_evidence=all_evidence,
            output_conclusion=synthesis,
            confidence=synthesis_confidence,
            reasoning_chain=f"Synthesized from {len(reasoning_steps)-1} reasoning steps"
        )
    
    def _calculate_overall_confidence(self, reasoning_steps: List[ReasoningStep]) -> float:
        """Calculate overall confidence from reasoning steps."""
        if not reasoning_steps:
            return 0.1
        
        # Weight later steps more heavily (they build on earlier ones)
        weights = [1.0 + 0.1 * i for i in range(len(reasoning_steps))]
        total_weight = sum(weights)
        
        weighted_confidence = sum(
            step.confidence * weight 
            for step, weight in zip(reasoning_steps, weights)
        )
        
        return weighted_confidence / total_weight
    
    async def _generate_alternatives(self, query_plan: QueryPlan, reasoning_steps: List[ReasoningStep]) -> List[str]:
        """Generate alternative explanations or answers."""
        alternatives = []
        
        # Look for lower-confidence reasoning steps that might suggest alternatives
        for step in reasoning_steps:
            if 0.3 <= step.confidence <= 0.6:  # Medium confidence suggests uncertainty
                alt_explanation = f"Alternative view: {step.output_conclusion[:100]}..."
                if alt_explanation not in alternatives:
                    alternatives.append(alt_explanation)
        
        # Generate methodological alternatives
        if query_plan.intent.primary_intent in [QueryType.ANALYTICAL, QueryType.COMPARATIVE]:
            alternatives.append("Alternative analytical approach: Different methodological framework might yield different insights")
        
        # Suggest need for more evidence if confidence is low
        overall_conf = self._calculate_overall_confidence(reasoning_steps)
        if overall_conf < 0.6:
            alternatives.append("Alternative explanation: Additional evidence needed to support or refute current conclusion")
        
        return alternatives[:3]  # Limit to top 3 alternatives
    
    async def _analyze_uncertainties(self, reasoning_steps: List[ReasoningStep], evidence: List[Evidence]) -> List[str]:
        """Analyze sources of uncertainty in reasoning."""
        uncertainties = []
        
        # Evidence quality issues
        low_conf_evidence = [ev for ev in evidence if ev.confidence < 0.6]
        if low_conf_evidence:
            uncertainties.append(f"Low confidence evidence: {len(low_conf_evidence)} sources have confidence below 0.6")
        
        # Reasoning step uncertainties
        uncertain_steps = [step for step in reasoning_steps if step.confidence < 0.7]
        if uncertain_steps:
            uncertainties.append(f"Uncertain reasoning steps: {len(uncertain_steps)} steps have lower confidence")
        
        # Coverage issues
        if len(evidence) < 3:
            uncertainties.append("Limited evidence: Few sources available for comprehensive analysis")
        
        # Conflicting evidence
        high_conf_evidence = [ev for ev in evidence if ev.confidence > 0.8]
        if len(high_conf_evidence) < len(evidence) / 2:
            uncertainties.append("Mixed evidence quality: Significant variation in source reliability")
        
        return uncertainties


# Convenience functions
async def create_reasoning_engine(vector_index: FAISSIndexManager,
                                graph_engine: CypherTraversalEngine) -> ReasoningEngine:
    """Create reasoning engine with provided components."""
    return ReasoningEngine(vector_index, graph_engine)


async def reason_about_query(query_plan: QueryPlan,
                           reasoning_engine: ReasoningEngine,
                           stream: bool = False) -> Union[ReasoningResult, AsyncGenerator]:
    """Execute reasoning for a query plan."""
    return await reasoning_engine.reason(query_plan, stream_results=stream)
#!/usr/bin/env python3
"""
Query Analyzer for Agentic Graph RAG
Intelligent query analysis, decomposition, and routing system.
"""

import logging
import re
import asyncio
from typing import List, Dict, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Types of queries the system can handle."""
    FACTUAL = "factual"                    # Direct fact retrieval
    CONCEPTUAL = "conceptual"              # Concept explanation
    RELATIONAL = "relational"             # Relationship queries
    ANALYTICAL = "analytical"             # Analysis and insights
    PROCEDURAL = "procedural"             # How-to questions
    COMPARATIVE = "comparative"           # Comparison questions
    CAUSAL = "causal"                     # Cause-effect questions
    TEMPORAL = "temporal"                 # Time-related questions
    SPATIAL = "spatial"                   # Location-related questions
    EXPLORATORY = "exploratory"          # Open-ended exploration


class QueryComplexity(Enum):
    """Query complexity levels."""
    SIMPLE = 1      # Single entity/relationship lookup
    MODERATE = 2    # Multiple entities or simple reasoning
    COMPLEX = 3     # Multi-hop reasoning or aggregation
    EXPERT = 4      # Deep analysis or complex inference


@dataclass
class QueryIntent:
    """Represents the intent behind a query."""
    primary_intent: QueryType
    secondary_intents: List[QueryType]
    confidence: float
    entities: List[str]
    relationships: List[str]
    keywords: List[str]
    question_words: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'primary_intent': self.primary_intent.value,
            'secondary_intents': [intent.value for intent in self.secondary_intents],
            'confidence': self.confidence,
            'entities': self.entities,
            'relationships': self.relationships,
            'keywords': self.keywords,
            'question_words': self.question_words
        }


@dataclass
class QueryDecomposition:
    """Decomposed query into sub-questions."""
    main_query: str
    sub_queries: List[str]
    dependencies: Dict[str, List[str]]  # Which sub-queries depend on others
    execution_order: List[str]
    complexity: QueryComplexity
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'main_query': self.main_query,
            'sub_queries': self.sub_queries,
            'dependencies': self.dependencies,
            'execution_order': self.execution_order,
            'complexity': self.complexity.value
        }


@dataclass
class QueryPlan:
    """Complete execution plan for a query."""
    query: str
    intent: QueryIntent
    decomposition: QueryDecomposition
    search_strategies: List[str]
    reasoning_steps: List[str]
    expected_sources: List[str]
    confidence_threshold: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'query': self.query,
            'intent': self.intent.to_dict(),
            'decomposition': self.decomposition.to_dict(),
            'search_strategies': self.search_strategies,
            'reasoning_steps': self.reasoning_steps,
            'expected_sources': self.expected_sources,
            'confidence_threshold': self.confidence_threshold
        }


class QueryAnalyzer:
    """
    Intelligent query analysis and planning system.
    
    Features:
    - Intent classification and confidence scoring
    - Named entity recognition for queries
    - Query decomposition into sub-tasks
    - Execution planning and optimization
    - Context-aware query rewriting
    - Multi-modal query support
    """
    
    def __init__(self):
        """Initialize query analyzer."""
        # Intent classification patterns
        self.intent_patterns = self._initialize_intent_patterns()
        
        # Entity extraction patterns
        self.entity_patterns = self._initialize_entity_patterns()
        
        # Question word mappings
        self.question_words = {
            'what': [QueryType.FACTUAL, QueryType.CONCEPTUAL],
            'who': [QueryType.FACTUAL, QueryType.RELATIONAL],
            'where': [QueryType.SPATIAL, QueryType.FACTUAL],
            'when': [QueryType.TEMPORAL, QueryType.FACTUAL],
            'why': [QueryType.CAUSAL, QueryType.ANALYTICAL],
            'how': [QueryType.PROCEDURAL, QueryType.ANALYTICAL],
            'which': [QueryType.COMPARATIVE, QueryType.FACTUAL],
            'whose': [QueryType.RELATIONAL, QueryType.FACTUAL],
            'compare': [QueryType.COMPARATIVE, QueryType.ANALYTICAL],
            'analyze': [QueryType.ANALYTICAL, QueryType.EXPLORATORY],
            'explain': [QueryType.CONCEPTUAL, QueryType.CAUSAL],
            'describe': [QueryType.CONCEPTUAL, QueryType.FACTUAL],
            'list': [QueryType.FACTUAL, QueryType.ANALYTICAL],
            'find': [QueryType.FACTUAL, QueryType.EXPLORATORY],
            'show': [QueryType.FACTUAL, QueryType.RELATIONAL]
        }
        
        # Relationship indicators
        self.relationship_indicators = [
            'related to', 'connected to', 'associated with', 'linked to',
            'caused by', 'results in', 'leads to', 'influences',
            'part of', 'belongs to', 'contains', 'includes',
            'similar to', 'different from', 'compared to',
            'before', 'after', 'during', 'while'
        ]
        
        logger.info("Query Analyzer initialized")
    
    def _initialize_intent_patterns(self) -> Dict[QueryType, List[str]]:
        """Initialize patterns for intent classification."""
        return {
            QueryType.FACTUAL: [
                r'\bwhat\s+is\b', r'\bwho\s+is\b', r'\bwhen\s+did\b',
                r'\bwhere\s+is\b', r'\bdefine\b', r'\btell\s+me\s+about\b',
                r'\blist\b', r'\benumerate\b', r'\bidentify\b'
            ],
            QueryType.CONCEPTUAL: [
                r'\bexplain\b', r'\bdescribe\b', r'\bwhat\s+does.*mean\b',
                r'\bhow\s+does.*work\b', r'\bwhat\s+is\s+the\s+concept\b',
                r'\bunderstand\b', r'\bclarify\b'
            ],
            QueryType.RELATIONAL: [
                r'\brelationship\s+between\b', r'\bconnection\s+between\b',
                r'\bhow.*related\b', r'\bassociated\s+with\b',
                r'\blinked\s+to\b', r'\bconnected\s+to\b'
            ],
            QueryType.ANALYTICAL: [
                r'\banalyze\b', r'\banalysis\b', r'\binsight\b',
                r'\bpattern\b', r'\btrend\b', r'\bstatistic\b',
                r'\bwhy\s+is\b', r'\breason\s+for\b', r'\bcause\s+of\b'
            ],
            QueryType.COMPARATIVE: [
                r'\bcompare\b', r'\bcomparison\b', r'\bdifference\s+between\b',
                r'\bsimilarity\s+between\b', r'\bversus\b', r'\bvs\b',
                r'\bbetter\s+than\b', r'\bworse\s+than\b'
            ],
            QueryType.CAUSAL: [
                r'\bwhy\s+did\b', r'\bwhy\s+does\b', r'\bcause\s+of\b',
                r'\breason\s+for\b', r'\bresult\s+of\b', r'\bleads\s+to\b',
                r'\bconsequence\b', r'\beffect\s+of\b'
            ],
            QueryType.TEMPORAL: [
                r'\bwhen\b', r'\bbefore\b', r'\bafter\b', r'\bduring\b',
                r'\btimeline\b', r'\bchronological\b', r'\bsequence\b',
                r'\bhistory\s+of\b', r'\bevolution\s+of\b'
            ],
            QueryType.PROCEDURAL: [
                r'\bhow\s+to\b', r'\bsteps\s+to\b', r'\bprocess\s+of\b',
                r'\bprocedure\b', r'\bmethod\b', r'\bway\s+to\b',
                r'\binstructions\b', r'\bguide\b'
            ]
        }
    
    def _initialize_entity_patterns(self) -> List[str]:
        """Initialize patterns for entity extraction."""
        return [
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',  # Proper nouns
            r'\b[A-Z]{2,}\b',  # Acronyms
            r'\d+(?:\.\d+)?',  # Numbers
            r'\b\d{4}\b',      # Years
            r'"[^"]+"',        # Quoted strings
            r"'[^']+'",        # Single quoted strings
        ]
    
    async def analyze_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> QueryPlan:
        """
        Analyze a query and create execution plan.
        
        Args:
            query: User query string
            context: Optional context from previous queries
            
        Returns:
            QueryPlan with intent, decomposition, and execution strategy
        """
        logger.debug(f"Analyzing query: {query}")
        
        # Clean and normalize query
        normalized_query = self._normalize_query(query)
        
        # Extract intent
        intent = await self._extract_intent(normalized_query)
        
        # Decompose query
        decomposition = await self._decompose_query(normalized_query, intent)
        
        # Generate execution strategies
        search_strategies = self._generate_search_strategies(intent, decomposition)
        reasoning_steps = self._generate_reasoning_steps(intent, decomposition)
        
        # Determine expected sources
        expected_sources = self._determine_expected_sources(intent, decomposition)
        
        # Set confidence threshold
        confidence_threshold = self._calculate_confidence_threshold(intent, decomposition)
        
        query_plan = QueryPlan(
            query=query,
            intent=intent,
            decomposition=decomposition,
            search_strategies=search_strategies,
            reasoning_steps=reasoning_steps,
            expected_sources=expected_sources,
            confidence_threshold=confidence_threshold
        )
        
        logger.info(f"Query analyzed: {intent.primary_intent.value} ({intent.confidence:.2f} confidence)")
        return query_plan
    
    def _normalize_query(self, query: str) -> str:
        """Normalize query text for analysis."""
        # Convert to lowercase for pattern matching
        normalized = query.lower().strip()
        
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Handle common contractions
        contractions = {
            "what's": "what is",
            "who's": "who is", 
            "where's": "where is",
            "when's": "when is",
            "why's": "why is",
            "how's": "how is",
            "can't": "cannot",
            "won't": "will not",
            "don't": "do not",
            "doesn't": "does not"
        }
        
        for contraction, expansion in contractions.items():
            normalized = normalized.replace(contraction, expansion)
        
        return normalized
    
    async def _extract_intent(self, query: str) -> QueryIntent:
        """Extract query intent using pattern matching and heuristics."""
        intent_scores = {}
        
        # Score based on patterns
        for intent_type, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    score += 1
            intent_scores[intent_type] = score
        
        # Score based on question words
        question_words_found = []
        for word, intents in self.question_words.items():
            if word in query:
                question_words_found.append(word)
                for intent in intents:
                    intent_scores[intent] = intent_scores.get(intent, 0) + 0.5
        
        # Find primary intent
        if not intent_scores:
            primary_intent = QueryType.EXPLORATORY
            confidence = 0.3
        else:
            primary_intent = max(intent_scores, key=intent_scores.get)
            max_score = intent_scores[primary_intent]
            total_score = sum(intent_scores.values())
            confidence = max_score / total_score if total_score > 0 else 0.3
        
        # Find secondary intents
        secondary_intents = [
            intent for intent, score in intent_scores.items()
            if intent != primary_intent and score > 0
        ]
        
        # Extract entities and relationships
        entities = self._extract_entities(query)
        relationships = self._extract_relationships(query)
        keywords = self._extract_keywords(query)
        
        return QueryIntent(
            primary_intent=primary_intent,
            secondary_intents=secondary_intents,
            confidence=confidence,
            entities=entities,
            relationships=relationships,
            keywords=keywords,
            question_words=question_words_found
        )
    
    def _extract_entities(self, query: str) -> List[str]:
        """Extract potential entities from query."""
        entities = []
        
        # Use entity patterns
        for pattern in self.entity_patterns:
            matches = re.findall(pattern, query)
            entities.extend(matches)
        
        # Clean and deduplicate
        cleaned_entities = []
        for entity in entities:
            entity = entity.strip(' "\'')
            if entity and len(entity) > 1 and entity not in cleaned_entities:
                cleaned_entities.append(entity)
        
        return cleaned_entities
    
    def _extract_relationships(self, query: str) -> List[str]:
        """Extract relationship indicators from query."""
        relationships = []
        
        for indicator in self.relationship_indicators:
            if indicator in query:
                relationships.append(indicator)
        
        return relationships
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract important keywords from query."""
        # Simple keyword extraction - remove stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'
        }
        
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        return keywords
    
    async def _decompose_query(self, query: str, intent: QueryIntent) -> QueryDecomposition:
        """Decompose complex queries into sub-questions."""
        sub_queries = []
        dependencies = {}
        
        # Simple decomposition based on intent
        if intent.primary_intent == QueryType.COMPARATIVE:
            # For comparative queries, create sub-queries for each entity
            if len(intent.entities) >= 2:
                for entity in intent.entities[:2]:  # Limit to first two
                    sub_queries.append(f"What are the properties of {entity}?")
                sub_queries.append(f"Compare {' and '.join(intent.entities[:2])}")
                
                # Set dependencies
                compare_query = sub_queries[-1]
                dependencies[compare_query] = sub_queries[:-1]
        
        elif intent.primary_intent == QueryType.CAUSAL:
            # For causal queries, find cause and effect
            if 'why' in query or 'cause' in query:
                sub_queries.append("Identify the main factors")
                sub_queries.append("Analyze the relationships")
                sub_queries.append("Determine causal links")
                
                # Sequential dependencies
                dependencies[sub_queries[1]] = [sub_queries[0]]
                dependencies[sub_queries[2]] = [sub_queries[1]]
        
        elif intent.primary_intent == QueryType.ANALYTICAL:
            # For analytical queries, break down analysis steps
            sub_queries.append("Gather relevant data")
            sub_queries.append("Identify patterns")
            sub_queries.append("Draw insights")
            
            # Sequential dependencies
            dependencies[sub_queries[1]] = [sub_queries[0]]
            dependencies[sub_queries[2]] = [sub_queries[1]]
        
        # If no sub-queries generated, treat as simple query
        if not sub_queries:
            sub_queries = [query]
        
        # Determine execution order
        execution_order = self._calculate_execution_order(sub_queries, dependencies)
        
        # Assess complexity
        complexity = self._assess_complexity(query, intent, sub_queries)
        
        return QueryDecomposition(
            main_query=query,
            sub_queries=sub_queries,
            dependencies=dependencies,
            execution_order=execution_order,
            complexity=complexity
        )
    
    def _calculate_execution_order(self, sub_queries: List[str], dependencies: Dict[str, List[str]]) -> List[str]:
        """Calculate optimal execution order based on dependencies."""
        # Simple topological sort
        remaining = set(sub_queries)
        execution_order = []
        
        while remaining:
            # Find queries with no unmet dependencies
            ready = []
            for query in remaining:
                deps = dependencies.get(query, [])
                if all(dep in execution_order for dep in deps):
                    ready.append(query)
            
            if not ready:
                # No dependencies or circular dependency - just add remaining
                ready = list(remaining)
            
            # Add first ready query
            next_query = ready[0]
            execution_order.append(next_query)
            remaining.remove(next_query)
        
        return execution_order
    
    def _assess_complexity(self, query: str, intent: QueryIntent, sub_queries: List[str]) -> QueryComplexity:
        """Assess query complexity based on various factors."""
        complexity_score = 1
        
        # Factor 1: Number of sub-queries
        if len(sub_queries) > 1:
            complexity_score += len(sub_queries) * 0.5
        
        # Factor 2: Number of entities
        if len(intent.entities) > 2:
            complexity_score += len(intent.entities) * 0.3
        
        # Factor 3: Intent type
        complex_intents = [QueryType.ANALYTICAL, QueryType.CAUSAL, QueryType.COMPARATIVE]
        if intent.primary_intent in complex_intents:
            complexity_score += 1
        
        # Factor 4: Query length (longer queries tend to be more complex)
        if len(query.split()) > 10:
            complexity_score += 0.5
        
        # Factor 5: Relationship indicators
        if len(intent.relationships) > 1:
            complexity_score += 0.5
        
        # Map to complexity enum
        if complexity_score <= 1.5:
            return QueryComplexity.SIMPLE
        elif complexity_score <= 2.5:
            return QueryComplexity.MODERATE
        elif complexity_score <= 3.5:
            return QueryComplexity.COMPLEX
        else:
            return QueryComplexity.EXPERT
    
    def _generate_search_strategies(self, intent: QueryIntent, decomposition: QueryDecomposition) -> List[str]:
        """Generate search strategies based on query analysis."""
        strategies = []
        
        # Based on intent
        if intent.primary_intent in [QueryType.FACTUAL, QueryType.CONCEPTUAL]:
            strategies.append("vector_similarity")
            strategies.append("entity_lookup")
        
        elif intent.primary_intent == QueryType.RELATIONAL:
            strategies.append("graph_traversal")
            strategies.append("relationship_search")
        
        elif intent.primary_intent == QueryType.ANALYTICAL:
            strategies.append("multi_hop_reasoning")
            strategies.append("pattern_matching")
        
        # Based on entities
        if intent.entities:
            strategies.append("entity_context_expansion")
        
        # Based on complexity
        if decomposition.complexity in [QueryComplexity.COMPLEX, QueryComplexity.EXPERT]:
            strategies.append("iterative_refinement")
            strategies.append("multi_source_aggregation")
        
        return list(set(strategies))  # Remove duplicates
    
    def _generate_reasoning_steps(self, intent: QueryIntent, decomposition: QueryDecomposition) -> List[str]:
        """Generate reasoning steps for query execution."""
        steps = []
        
        # Standard reasoning flow
        steps.append("Parse and understand query")
        steps.append("Identify key entities and concepts")
        
        # Intent-specific reasoning
        if intent.primary_intent == QueryType.FACTUAL:
            steps.extend([
                "Search for direct facts",
                "Verify information consistency",
                "Compile factual response"
            ])
        
        elif intent.primary_intent == QueryType.ANALYTICAL:
            steps.extend([
                "Gather relevant data points",
                "Identify patterns and connections", 
                "Apply analytical reasoning",
                "Generate insights and conclusions"
            ])
        
        elif intent.primary_intent == QueryType.COMPARATIVE:
            steps.extend([
                "Collect information about each entity",
                "Identify comparison dimensions",
                "Analyze similarities and differences",
                "Synthesize comparative analysis"
            ])
        
        # Complexity-based steps
        if decomposition.complexity in [QueryComplexity.COMPLEX, QueryComplexity.EXPERT]:
            steps.append("Apply multi-step reasoning")
            steps.append("Validate reasoning chain")
        
        steps.append("Generate final response")
        
        return steps
    
    def _determine_expected_sources(self, intent: QueryIntent, decomposition: QueryDecomposition) -> List[str]:
        """Determine expected information sources."""
        sources = []
        
        # Based on intent
        if intent.primary_intent in [QueryType.FACTUAL, QueryType.CONCEPTUAL]:
            sources.extend(["knowledge_graph", "vector_embeddings"])
        
        elif intent.primary_intent == QueryType.RELATIONAL:
            sources.extend(["graph_relationships", "entity_connections"])
        
        elif intent.primary_intent == QueryType.ANALYTICAL:
            sources.extend(["aggregated_data", "pattern_analysis", "inference_chains"])
        
        # Based on entities
        if intent.entities:
            sources.append("entity_profiles")
        
        # Default sources
        if not sources:
            sources = ["knowledge_graph", "vector_embeddings"]
        
        return sources
    
    def _calculate_confidence_threshold(self, intent: QueryIntent, decomposition: QueryDecomposition) -> float:
        """Calculate confidence threshold for response validation."""
        base_threshold = 0.7
        
        # Adjust based on complexity
        if decomposition.complexity == QueryComplexity.SIMPLE:
            return base_threshold - 0.1
        elif decomposition.complexity == QueryComplexity.EXPERT:
            return base_threshold + 0.1
        
        # Adjust based on intent confidence
        if intent.confidence < 0.5:
            return base_threshold + 0.1  # Higher threshold for uncertain intents
        
        return base_threshold
    
    async def refine_query(self, query: str, feedback: Dict[str, Any]) -> str:
        """Refine query based on feedback or partial results."""
        # Simple query refinement based on feedback
        if feedback.get('missing_entities'):
            # Add missing entities to query
            missing = ', '.join(feedback['missing_entities'])
            refined_query = f"{query} including {missing}"
            return refined_query
        
        if feedback.get('too_broad'):
            # Add constraints to narrow down query
            refined_query = f"{query} specifically"
            return refined_query
        
        if feedback.get('no_results'):
            # Generalize query
            # Remove specific terms and use broader language
            refined_query = re.sub(r'\bspecific\b|\bexact\b|\bprecise\b', '', query)
            return refined_query.strip()
        
        return query
    
    def get_query_suggestions(self, partial_query: str) -> List[str]:
        """Generate query suggestions based on partial input."""
        suggestions = []
        
        partial_lower = partial_query.lower()
        
        # Intent-based suggestions
        if partial_lower.startswith('what'):
            suggestions.extend([
                "What is the relationship between...",
                "What are the properties of...",
                "What causes...",
                "What is the difference between..."
            ])
        
        elif partial_lower.startswith('how'):
            suggestions.extend([
                "How are ... connected?",
                "How does ... work?",
                "How to ...",
                "How is ... related to ..."
            ])
        
        elif partial_lower.startswith('why'):
            suggestions.extend([
                "Why does ... happen?",
                "Why is ... important?",
                "Why are ... connected?"
            ])
        
        # Generic suggestions
        if len(suggestions) < 3:
            suggestions.extend([
                "Find entities related to...",
                "Explain the concept of...",
                "Compare ... and ...",
                "Analyze the pattern in..."
            ])
        
        return suggestions[:5]  # Limit to 5 suggestions


# Convenience functions
async def analyze_user_query(query: str, context: Optional[Dict[str, Any]] = None) -> QueryPlan:
    """Analyze user query and return execution plan."""
    analyzer = QueryAnalyzer()
    return await analyzer.analyze_query(query, context)


def extract_query_entities(query: str) -> List[str]:
    """Extract entities from query string."""
    analyzer = QueryAnalyzer()
    return analyzer._extract_entities(query)
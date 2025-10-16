#!/usr/bin/env python3
"""
LLM Ontology Generator for Agentic Graph RAG Pipeline
Uses OpenRouter API to generate structured ontologies from raw text.
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import re
import os

import httpx
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class EntitySchema(BaseModel):
    """Schema for extracted entities."""
    name: str = Field(description="Entity name/identifier")
    type: str = Field(description="Entity type (person, organization, concept, etc.)")
    description: str = Field(description="Brief description of the entity")
    confidence: float = Field(description="Confidence score (0.0-1.0)")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Additional properties")


class RelationshipSchema(BaseModel):
    """Schema for extracted relationships."""
    source_entity: str = Field(description="Source entity name")
    target_entity: str = Field(description="Target entity name")
    relationship_type: str = Field(description="Type of relationship")
    description: str = Field(description="Description of the relationship")
    confidence: float = Field(description="Confidence score (0.0-1.0)")
    evidence: str = Field(description="Text evidence supporting this relationship")


class OntologySchema(BaseModel):
    """Complete ontology schema."""
    entities: List[EntitySchema] = Field(default_factory=list)
    relationships: List[RelationshipSchema] = Field(default_factory=list)
    entity_types: List[str] = Field(default_factory=list)
    relationship_types: List[str] = Field(default_factory=list)
    domain: str = Field(description="Domain or topic of the ontology")
    confidence: float = Field(description="Overall ontology confidence")


@dataclass
class LLMConfig:
    """Configuration for LLM API."""
    api_key: str
    base_url: str = "https://openrouter.ai/api/v1"
    model: str = "microsoft/wizardlm-2-8x22b"
    max_tokens: int = 4000
    temperature: float = 0.1
    timeout: int = 30


class LLMOntologyGenerator:
    """
    Advanced ontology generator using Large Language Models.
    
    Features:
    - Structured entity and relationship extraction
    - Domain-specific ontology generation
    - Confidence scoring and evidence tracking
    - Hierarchical entity relationships
    - Multi-pass refinement and validation
    """
    
    def __init__(self, config: LLMConfig):
        """Initialize LLM ontology generator."""
        self.config = config
        self.client = httpx.AsyncClient(
            base_url=config.base_url,
            headers={
                "Authorization": f"Bearer {config.api_key}",
                "HTTP-Referer": "https://github.com/agentic-graph-rag",
                "X-Title": "Agentic Graph RAG Pipeline"
            },
            timeout=config.timeout
        )
        
        # Ontology prompts
        self.entity_extraction_prompt = self._load_entity_extraction_prompt()
        self.relationship_extraction_prompt = self._load_relationship_extraction_prompt()
        self.ontology_refinement_prompt = self._load_ontology_refinement_prompt()
    
    def _load_entity_extraction_prompt(self) -> str:
        """Load entity extraction prompt template."""
        return """
You are an expert knowledge engineer specializing in ontology construction. Your task is to extract entities from text and create a structured representation.

**Instructions:**
1. Identify all significant entities in the text (persons, organizations, concepts, locations, events, etc.)
2. Classify each entity into appropriate types
3. Provide confidence scores based on contextual evidence
4. Extract key properties and attributes

**Entity Types to Consider:**
- Person: Individual people mentioned
- Organization: Companies, institutions, groups
- Concept: Ideas, theories, methodologies, systems
- Location: Places, geographical entities
- Event: Occurrences, processes, activities
- Technology: Tools, software, hardware
- Document: Publications, papers, reports
- Temporal: Time periods, dates, durations

**Output Format:**
Return a JSON object with the following structure:
{
    "entities": [
        {
            "name": "entity name",
            "type": "entity_type",
            "description": "brief description",
            "confidence": 0.95,
            "properties": {
                "additional_key": "value"
            }
        }
    ],
    "entity_types": ["list of unique entity types found"]
}

**Text to Analyze:**
{text}

**Response (JSON only):**
"""
    
    def _load_relationship_extraction_prompt(self) -> str:
        """Load relationship extraction prompt template."""
        return """
You are an expert in knowledge graph construction. Extract meaningful relationships between entities from the given text.

**Instructions:**
1. Identify relationships between the entities provided
2. Determine the type and direction of each relationship
3. Provide evidence from the text supporting each relationship
4. Score confidence based on textual evidence strength

**Relationship Types to Consider:**
- works_for: Employment, affiliation
- located_in: Geographical relationships
- part_of: Hierarchical containment
- collaborates_with: Partnerships, cooperation
- develops: Creation, development relationships
- uses: Utilization, application
- influences: Impact, effect relationships
- similar_to: Similarity, equivalence
- precedes: Temporal relationships
- causes: Causal relationships

**Known Entities:**
{entities}

**Output Format:**
{
    "relationships": [
        {
            "source_entity": "source entity name",
            "target_entity": "target entity name", 
            "relationship_type": "relationship_type",
            "description": "relationship description",
            "confidence": 0.85,
            "evidence": "supporting text from document"
        }
    ],
    "relationship_types": ["list of unique relationship types"]
}

**Text to Analyze:**
{text}

**Response (JSON only):**
"""
    
    def _load_ontology_refinement_prompt(self) -> str:
        """Load ontology refinement prompt template."""
        return """
You are an ontology expert. Review and refine the extracted entities and relationships to create a coherent knowledge graph.

**Instructions:**
1. Resolve entity duplicates and variations
2. Standardize entity and relationship types
3. Add missing implied relationships
4. Improve descriptions and confidence scores
5. Ensure ontological consistency

**Current Ontology:**
{ontology}

**Refinement Tasks:**
- Merge duplicate or similar entities
- Standardize naming conventions
- Add hierarchical relationships (is_a, part_of)
- Resolve ambiguous references
- Improve type classifications

**Output Format:**
{
    "refined_entities": [...],
    "refined_relationships": [...],
    "entity_types": [...],
    "relationship_types": [...],
    "refinement_notes": "summary of changes made"
}

**Response (JSON only):**
"""
    
    async def extract_ontology(self, 
                              text: str, 
                              domain_hint: Optional[str] = None,
                              existing_ontology: Optional[OntologySchema] = None) -> OntologySchema:
        """
        Extract comprehensive ontology from text using multi-pass LLM processing.
        
        Args:
            text: Input text to analyze
            domain_hint: Optional domain context
            existing_ontology: Previous ontology to extend
            
        Returns:
            Complete ontology schema
        """
        logger.info(f"Extracting ontology from {len(text)} characters of text")
        
        try:
            # Step 1: Entity Extraction
            entities = await self._extract_entities(text, domain_hint)
            logger.info(f"Extracted {len(entities)} entities")
            
            # Step 2: Relationship Extraction
            relationships = await self._extract_relationships(text, entities)
            logger.info(f"Extracted {len(relationships)} relationships")
            
            # Step 3: Ontology Refinement
            refined_ontology = await self._refine_ontology(entities, relationships, text)
            logger.info("Ontology refinement completed")
            
            # Step 4: Build final schema
            ontology = OntologySchema(
                entities=refined_ontology.get('refined_entities', entities),
                relationships=refined_ontology.get('refined_relationships', relationships),
                entity_types=refined_ontology.get('entity_types', []),
                relationship_types=refined_ontology.get('relationship_types', []),
                domain=domain_hint or self._infer_domain(text),
                confidence=self._calculate_overall_confidence(entities, relationships)
            )
            
            return ontology
            
        except Exception as e:
            logger.error(f"Ontology extraction failed: {e}")
            # Return minimal ontology
            return OntologySchema(
                entities=[],
                relationships=[],
                entity_types=[],
                relationship_types=[],
                domain="unknown",
                confidence=0.0
            )
    
    async def _extract_entities(self, text: str, domain_hint: Optional[str] = None) -> List[EntitySchema]:
        """Extract entities using LLM."""
        prompt = self.entity_extraction_prompt.format(text=text[:4000])  # Limit text length
        
        try:
            response = await self._call_llm(prompt)
            
            # Parse JSON response
            result = json.loads(response)
            entities = []
            
            for entity_data in result.get('entities', []):
                try:
                    entity = EntitySchema(
                        name=entity_data['name'],
                        type=entity_data['type'],
                        description=entity_data.get('description', ''),
                        confidence=entity_data.get('confidence', 0.5),
                        properties=entity_data.get('properties', {})
                    )
                    entities.append(entity)
                except Exception as e:
                    logger.warning(f"Invalid entity data: {entity_data} - {e}")
            
            return entities
            
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return []
    
    async def _extract_relationships(self, text: str, entities: List[EntitySchema]) -> List[RelationshipSchema]:
        """Extract relationships using LLM."""
        # Prepare entity list for prompt
        entity_names = [entity.name for entity in entities]
        entities_text = "\n".join(f"- {name}" for name in entity_names[:50])  # Limit entities
        
        prompt = self.relationship_extraction_prompt.format(
            text=text[:3000],  # Limit text length
            entities=entities_text
        )
        
        try:
            response = await self._call_llm(prompt)
            
            # Parse JSON response
            result = json.loads(response)
            relationships = []
            
            for rel_data in result.get('relationships', []):
                try:
                    relationship = RelationshipSchema(
                        source_entity=rel_data['source_entity'],
                        target_entity=rel_data['target_entity'],
                        relationship_type=rel_data['relationship_type'],
                        description=rel_data.get('description', ''),
                        confidence=rel_data.get('confidence', 0.5),
                        evidence=rel_data.get('evidence', '')
                    )
                    relationships.append(relationship)
                except Exception as e:
                    logger.warning(f"Invalid relationship data: {rel_data} - {e}")
            
            return relationships
            
        except Exception as e:
            logger.error(f"Relationship extraction failed: {e}")
            return []
    
    async def _refine_ontology(self, 
                              entities: List[EntitySchema], 
                              relationships: List[RelationshipSchema],
                              original_text: str) -> Dict[str, Any]:
        """Refine and improve ontology using LLM."""
        
        # Create current ontology representation
        ontology_dict = {
            "entities": [entity.model_dump() for entity in entities],
            "relationships": [rel.model_dump() for rel in relationships],
            "entity_count": len(entities),
            "relationship_count": len(relationships)
        }
        
        prompt = self.ontology_refinement_prompt.format(
            ontology=json.dumps(ontology_dict, indent=2)
        )
        
        try:
            response = await self._call_llm(prompt)
            result = json.loads(response)
            
            # Parse refined entities
            refined_entities = []
            for entity_data in result.get('refined_entities', []):
                try:
                    entity = EntitySchema(**entity_data)
                    refined_entities.append(entity)
                except Exception as e:
                    logger.warning(f"Invalid refined entity: {entity_data} - {e}")
            
            # Parse refined relationships
            refined_relationships = []
            for rel_data in result.get('refined_relationships', []):
                try:
                    relationship = RelationshipSchema(**rel_data)
                    refined_relationships.append(relationship)
                except Exception as e:
                    logger.warning(f"Invalid refined relationship: {rel_data} - {e}")
            
            return {
                'refined_entities': refined_entities or entities,
                'refined_relationships': refined_relationships or relationships,
                'entity_types': result.get('entity_types', []),
                'relationship_types': result.get('relationship_types', []),
                'refinement_notes': result.get('refinement_notes', '')
            }
            
        except Exception as e:
            logger.error(f"Ontology refinement failed: {e}")
            return {
                'refined_entities': entities,
                'refined_relationships': relationships,
                'entity_types': list(set(e.type for e in entities)),
                'relationship_types': list(set(r.relationship_type for r in relationships))
            }
    
    async def _call_llm(self, prompt: str) -> str:
        """Make API call to LLM service."""
        try:
            response = await self.client.post("/chat/completions", json={
                "model": self.config.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert ontology engineer. Always respond with valid JSON only, no additional text or formatting."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "response_format": {"type": "json_object"}
            })
            
            response.raise_for_status()
            result = response.json()
            
            if 'choices' in result and result['choices']:
                content = result['choices'][0]['message']['content']
                return content.strip()
            else:
                raise Exception(f"Unexpected API response: {result}")
                
        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
            raise
    
    def _infer_domain(self, text: str) -> str:
        """Infer domain from text content."""
        # Simple domain inference based on keywords
        domain_keywords = {
            'technology': ['software', 'system', 'algorithm', 'data', 'computer', 'digital'],
            'business': ['company', 'market', 'customer', 'product', 'revenue', 'strategy'],
            'science': ['research', 'study', 'experiment', 'hypothesis', 'theory', 'analysis'],
            'education': ['learning', 'student', 'curriculum', 'education', 'teaching', 'academic'],
            'healthcare': ['patient', 'treatment', 'medical', 'health', 'clinical', 'diagnosis']
        }
        
        text_lower = text.lower()
        domain_scores = {}
        
        for domain, keywords in domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            domain_scores[domain] = score
        
        if domain_scores and max(domain_scores.values()) > 0:
            return max(domain_scores, key=domain_scores.get)
        
        return "general"
    
    def _calculate_overall_confidence(self, 
                                    entities: List[EntitySchema], 
                                    relationships: List[RelationshipSchema]) -> float:
        """Calculate overall ontology confidence score."""
        if not entities and not relationships:
            return 0.0
        
        entity_confidences = [e.confidence for e in entities] if entities else [0.0]
        rel_confidences = [r.confidence for r in relationships] if relationships else [0.0]
        
        all_confidences = entity_confidences + rel_confidences
        return sum(all_confidences) / len(all_confidences)
    
    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()


# Utility functions for ontology processing
def ontology_to_graph_format(ontology: OntologySchema) -> Dict[str, Any]:
    """Convert ontology schema to graph visualization format."""
    
    # Convert entities
    entities = []
    entity_id_map = {}
    
    for i, entity in enumerate(ontology.entities):
        entity_id = f"entity_{i}"
        entity_id_map[entity.name] = entity_id
        
        entities.append({
            'id': entity_id,
            'name': entity.name,
            'type': entity.type,
            'description': entity.description,
            'confidence': entity.confidence,
            'properties': entity.properties
        })
    
    # Convert relationships
    relationships = []
    
    for i, rel in enumerate(ontology.relationships):
        source_id = entity_id_map.get(rel.source_entity)
        target_id = entity_id_map.get(rel.target_entity)
        
        if source_id and target_id:
            relationships.append({
                'id': f"rel_{i}",
                'source': source_id,
                'target': target_id,
                'type': rel.relationship_type,
                'description': rel.description,
                'confidence': rel.confidence,
                'evidence': rel.evidence
            })
    
    return {
        'entities': entities,
        'relationships': relationships,
        'metadata': {
            'domain': ontology.domain,
            'confidence': ontology.confidence,
            'entity_types': ontology.entity_types,
            'relationship_types': ontology.relationship_types
        }
    }


def load_llm_config_from_env() -> LLMConfig:
    """Load LLM configuration from environment variables."""
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not set")
    
    return LLMConfig(
        api_key=api_key,
        base_url=os.getenv('OPENROUTER_BASE_URL', 'https://openrouter.ai/api/v1'),
        model=os.getenv('OPENROUTER_MODEL', 'microsoft/wizardlm-2-8x22b'),
        max_tokens=int(os.getenv('OPENROUTER_MAX_TOKENS', '4000')),
        temperature=float(os.getenv('OPENROUTER_TEMPERATURE', '0.1'))
    )


# Test and demo functions
async def demo_ontology_generation():
    """Demonstration of ontology generation."""
    sample_text = """
    John Smith works at Microsoft as a Senior Software Engineer in the Azure AI team. 
    He collaborates with Sarah Johnson, who is a Product Manager for machine learning services.
    Microsoft Azure provides cloud computing services including artificial intelligence and machine learning capabilities.
    The Azure AI team develops cognitive services that help businesses build intelligent applications.
    John has published several research papers on natural language processing and computer vision.
    """
    
    try:
        config = load_llm_config_from_env()
        generator = LLMOntologyGenerator(config)
        
        print("Generating ontology from sample text...")
        ontology = await generator.extract_ontology(sample_text, domain_hint="technology")
        
        print(f"\nExtracted Ontology:")
        print(f"Domain: {ontology.domain}")
        print(f"Confidence: {ontology.confidence:.2f}")
        print(f"Entities: {len(ontology.entities)}")
        print(f"Relationships: {len(ontology.relationships)}")
        
        print("\nEntities:")
        for entity in ontology.entities:
            print(f"  - {entity.name} ({entity.type}) - {entity.confidence:.2f}")
        
        print("\nRelationships:")
        for rel in ontology.relationships:
            print(f"  - {rel.source_entity} --{rel.relationship_type}--> {rel.target_entity} ({rel.confidence:.2f})")
        
        await generator.close()
        
    except Exception as e:
        print(f"Demo failed: {e}")
        print("Make sure to set OPENROUTER_API_KEY environment variable")


if __name__ == "__main__":
    asyncio.run(demo_ontology_generation())
"""
Prompt management module for DocQA AI system.
Provides support for custom prompts, templating, versioning, and A/B testing.
"""

import os
import json
import re
import logging
from typing import Dict, Any, Optional, List, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
import hashlib
import yaml

from src.utils.logger import get_logger
from src.utils.cache import CacheManager, cached

logger = get_logger(__name__)


class PromptType(Enum):
    """Types of prompts."""
    QA = "qa"
    QA_WITH_CONTEXT = "qa_with_context"
    SUMMARIZATION = "summarization"
    CONVERSATION = "conversation"
    CLASSIFICATION = "classification"
    EXTRACTION = "extraction"
    CODE_GENERATION = "code_generation"
    REWRITING = "rewriting"
    TRANSLATION = "translation"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    ENTITY_RECOGNITION = "entity_recognition"
    HALLUCINATION_DETECTION = "hallucination_detection"
    QUERY_REWRITING = "query_rewriting"
    RESPONSE_RERANKING = "response_reranking"
    CUSTOM = "custom"


@dataclass
class PromptTemplate:
    """Prompt template definition."""
    name: str
    template: str
    prompt_type: PromptType
    description: str = ""
    version: str = "1.0.0"
    variables: List[str] = field(default_factory=list)
    examples: List[Dict[str, str]] = field(default_factory=list)
    system_prompt: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    is_active: bool = True
    tags: List[str] = field(default_factory=list)
    author: str = "system"
    success_rate: float = 0.0
    usage_count: int = 0
    average_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "template": self.template,
            "type": self.prompt_type.value,
            "description": self.description,
            "version": self.version,
            "variables": self.variables,
            "examples": self.examples,
            "system_prompt": self.system_prompt,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "is_active": self.is_active,
            "tags": self.tags,
            "author": self.author,
            "success_rate": self.success_rate,
            "usage_count": self.usage_count,
            "average_score": self.average_score
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PromptTemplate':
        """Create from dictionary."""
        return cls(
            name=data["name"],
            template=data["template"],
            prompt_type=PromptType(data["type"]),
            description=data.get("description", ""),
            version=data.get("version", "1.0.0"),
            variables=data.get("variables", []),
            examples=data.get("examples", []),
            system_prompt=data.get("system_prompt"),
            metadata=data.get("metadata", {}),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if "updated_at" in data else datetime.now(),
            is_active=data.get("is_active", True),
            tags=data.get("tags", []),
            author=data.get("author", "system"),
            success_rate=data.get("success_rate", 0.0),
            usage_count=data.get("usage_count", 0),
            average_score=data.get("average_score", 0.0)
        )

    def format(self, **kwargs) -> str:
        """Format the template with variables."""
        missing_vars = [var for var in self.variables if var not in kwargs]
        if missing_vars:
            raise ValueError(f"Missing required variables: {missing_vars}")

        try:
            return self.template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Unknown variable in template: {e}")

    def get_hash(self) -> str:
        """Get hash of the template for versioning."""
        content = f"{self.template}:{self.system_prompt or ''}"
        return hashlib.md5(content.encode()).hexdigest()


class PromptStorage:
    """
    Storage backend for prompt templates.
    Supports file system, memory, and database storage.
    """

    def __init__(
        self,
        storage_type: str = "memory",
        storage_path: Optional[str] = None,
        auto_save: bool = True
    ):
        """
        Initialize prompt storage.

        Args:
            storage_type: 'memory', 'file', 'database'
            storage_path: Path for file storage
            auto_save: Whether to auto-save changes
        """
        self.storage_type = storage_type
        self.storage_path = Path(storage_path) if storage_path else Path("./prompts")
        self.auto_save = auto_save

        self._prompts: Dict[str, PromptTemplate] = {}
        self._prompts_by_type: Dict[PromptType, List[str]] = {}
        self._initialized = False

        # Initialize storage
        self._initialize()

        logger.info(f"PromptStorage initialized: type={storage_type}, path={self.storage_path}")

    def _initialize(self):
        """Initialize storage backend."""
        if self.storage_type == "memory":
            self._load_default_prompts()
        elif self.storage_type == "file":
            self.storage_path.mkdir(parents=True, exist_ok=True)
            self._load_from_files()
        elif self.storage_type == "database":
            # Database initialization would go here
            self._load_default_prompts()
        else:
            raise ValueError(f"Unsupported storage type: {self.storage_type}")

        self._initialized = True

    def _load_default_prompts(self):
        """Load default prompt templates."""
        default_prompts = self._get_default_prompts()
        for prompt in default_prompts:
            self._prompts[prompt.name] = prompt
            if prompt.prompt_type not in self._prompts_by_type:
                self._prompts_by_type[prompt.prompt_type] = []
            if prompt.name not in self._prompts_by_type[prompt.prompt_type]:
                self._prompts_by_type[prompt.prompt_type].append(prompt.name)

    def _get_default_prompts(self) -> List[PromptTemplate]:
        """Get default prompt templates."""
        return [
            # QA prompts
            PromptTemplate(
                name="qa_basic",
                template="Question: {question}\n\nAnswer:",
                prompt_type=PromptType.QA,
                description="Basic QA without context",
                variables=["question"]
            ),
            PromptTemplate(
                name="qa_with_context",
                template="""You are a helpful assistant that answers questions based on the provided context.

Context:
{context}

Question: {question}

Instructions:
- Answer based ONLY on the information in the context
- If the answer cannot be found in the context, say "I cannot find this information in the provided documents"
- Be concise and specific
- Include relevant quotes when helpful

Answer:""",
                prompt_type=PromptType.QA_WITH_CONTEXT,
                description="QA with document context",
                variables=["context", "question"],
                system_prompt="You are a helpful assistant that answers questions based on provided documents."
            ),
            PromptTemplate(
                name="qa_with_sources",
                template="""You are an AI assistant answering questions based on provided document excerpts.

Documents:
{sources}

Question: {question}

Provide a comprehensive answer and cite which documents support your answer.
If information is not available in the documents, clearly state that.

Format your response as:
Answer: [your answer]
Sources: [list document IDs or names]

Answer:""",
                prompt_type=PromptType.QA_WITH_CONTEXT,
                description="QA with source attribution",
                variables=["sources", "question"]
            ),

            # Summarization prompts
            PromptTemplate(
                name="summarization_concise",
                template="""Summarize the following text in 2-3 sentences:

Text: {text}

Concise Summary:""",
                prompt_type=PromptType.SUMMARIZATION,
                description="Concise summarization",
                variables=["text"]
            ),
            PromptTemplate(
                name="summarization_detailed",
                template="""Provide a detailed summary of the following text, including key points and main arguments.

Text: {text}

Detailed Summary:""",
                prompt_type=PromptType.SUMMARIZATION,
                description="Detailed summarization with key points",
                variables=["text"]
            ),

            # Conversation prompts
            PromptTemplate(
                name="conversation_system",
                template="""You are a helpful AI assistant for document question answering.
Your responses should be:
- Accurate and based on provided information
- Clear and concise
- Professional but friendly

Context from documents:
{context}

Conversation history:
{history}

Current question: {question}

Provide a helpful response:""",
                prompt_type=PromptType.CONVERSATION,
                description="System prompt for conversational QA",
                variables=["context", "history", "question"]
            ),

            # Extraction prompts
            PromptTemplate(
                name="extraction_key_phrases",
                template="""Extract the most important key phrases from the following text.
Return them as a comma-separated list.

Text: {text}

Key phrases:""",
                prompt_type=PromptType.EXTRACTION,
                description="Extract key phrases",
                variables=["text"]
            ),
            PromptTemplate(
                name="extraction_entities",
                template="""Extract the following entity types from the text:
- Person names
- Organizations
- Dates
- Locations
- Monetary values

Text: {text}

Return in JSON format:
{
  "persons": [],
  "organizations": [],
  "dates": [],
  "locations": [],
  "monetary_values": []
}

Entities:""",
                prompt_type=PromptType.ENTITY_RECOGNITION,
                description="Extract named entities",
                variables=["text"]
            ),

            # RAG-specific prompts
            PromptTemplate(
                name="rag_response_generation",
                template="""You are a document QA assistant. Generate a response based on the retrieved context.

Retrieved Context:
{context}

User Question: {question}

Instructions:
1. Answer based ONLY on the provided context
2. If the context doesn't contain the answer, say "I cannot answer this based on the available documents"
3. Be accurate and cite specific parts of the context when relevant
4. Keep the response concise but comprehensive

Response:""",
                prompt_type=PromptType.QA_WITH_CONTEXT,
                description="Generate response from retrieved chunks",
                variables=["context", "question"]
            ),
            PromptTemplate(
                name="rag_hallucination_check",
                template="""Check if the following answer is supported by the provided context.

Context: {context}

Answer: {answer}

Is the answer fully supported by the context?
Respond with:
- "SUPPORTED" if the answer is directly stated or clearly implied
- "PARTIALLY_SUPPORTED" if only part of the answer is supported
- "UNSUPPORTED" if the answer contains information not in the context
- "CONTRADICTORY" if the answer contradicts the context

Then provide a brief explanation.

Assessment:""",
                prompt_type=PromptType.HALLUCINATION_DETECTION,
                description="Detect hallucinations in responses",
                variables=["context", "answer"]
            ),
            PromptTemplate(
                name="rag_response_reranking",
                template="""Given the question, rate the relevance of each candidate answer on a scale of 1-10.

Question: {question}

Candidate Answers:
{candidates}

For each candidate, provide:
- Relevance score (1-10)
- Brief justification

Output format:
Candidate 0: Score X/10 - Justification
Candidate 1: Score X/10 - Justification

Scores:""",
                prompt_type=PromptType.RESPONSE_RERANKING,
                description="Rerank candidate responses",
                variables=["question", "candidates"]
            ),

            # Query rewriting
            PromptTemplate(
                name="query_rewrite",
                template="""Given the conversation history, rewrite the follow-up question to be self-contained and clear.

Conversation history:
{history}

Follow-up question: {question}

Rewritten question:""",
                prompt_type=PromptType.QUERY_REWRITING,
                description="Rewrite queries for better retrieval",
                variables=["history", "question"]
            )
        ]

    def _load_from_files(self):
        """Load prompts from files."""
        # Load YAML files
        for file_path in self.storage_path.glob("*.yaml"):
            try:
                with open(file_path, 'r') as f:
                    data = yaml.safe_load(f)
                if isinstance(data, dict):
                    self._add_from_dict(data)
            except Exception as e:
                logger.warning(f"Failed to load prompt from {file_path}: {e}")

        # Load JSON files
        for file_path in self.storage_path.glob("*.json"):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    self._add_from_dict(data)
                elif isinstance(data, list):
                    for item in data:
                        self._add_from_dict(item)
            except Exception as e:
                logger.warning(f"Failed to load prompt from {file_path}: {e}")

        # Also load default prompts if no custom prompts found
        if not self._prompts:
            self._load_default_prompts()

    def _add_from_dict(self, data: Dict[str, Any]):
        """Add prompt from dictionary."""
        try:
            prompt = PromptTemplate.from_dict(data)
            self._prompts[prompt.name] = prompt
            if prompt.prompt_type not in self._prompts_by_type:
                self._prompts_by_type[prompt.prompt_type] = []
            if prompt.name not in self._prompts_by_type[prompt.prompt_type]:
                self._prompts_by_type[prompt.prompt_type].append(prompt.name)
        except Exception as e:
            logger.warning(f"Failed to add prompt: {e}")

    def save(self, prompt: PromptTemplate):
        """
        Save a prompt template.

        Args:
            prompt: PromptTemplate object
        """
        self._prompts[prompt.name] = prompt

        if prompt.prompt_type not in self._prompts_by_type:
            self._prompts_by_type[prompt.prompt_type] = []
        if prompt.name not in self._prompts_by_type[prompt.prompt_type]:
            self._prompts_by_type[prompt.prompt_type].append(prompt.name)

        if self.auto_save and self.storage_type == "file":
            self._save_to_file(prompt)

        logger.info(f"Saved prompt: {prompt.name}")

    def _save_to_file(self, prompt: PromptTemplate):
        """Save prompt to file."""
        file_path = self.storage_path / f"{prompt.name}.json"
        with open(file_path, 'w') as f:
            json.dump(prompt.to_dict(), f, indent=2)

    def get(self, name: str) -> Optional[PromptTemplate]:
        """Get a prompt by name."""
        return self._prompts.get(name)

    def get_by_type(self, prompt_type: PromptType) -> List[PromptTemplate]:
        """Get prompts by type."""
        names = self._prompts_by_type.get(prompt_type, [])
        return [self._prompts[name] for name in names if name in self._prompts]

    def list_prompts(
        self,
        prompt_type: Optional[PromptType] = None,
        active_only: bool = True,
        tags: Optional[List[str]] = None
    ) -> List[PromptTemplate]:
        """List prompts with filters."""
        prompts = list(self._prompts.values())

        if active_only:
            prompts = [p for p in prompts if p.is_active]

        if prompt_type:
            prompts = [p for p in prompts if p.prompt_type == prompt_type]

        if tags:
            prompts = [p for p in prompts if any(tag in p.tags for tag in tags)]

        return prompts

    def delete(self, name: str) -> bool:
        """Delete a prompt."""
        if name in self._prompts:
            prompt = self._prompts[name]
            del self._prompts[name]

            # Remove from type index
            if prompt.prompt_type in self._prompts_by_type:
                if name in self._prompts_by_type[prompt.prompt_type]:
                    self._prompts_by_type[prompt.prompt_type].remove(name)

            # Delete file
            if self.storage_type == "file":
                file_path = self.storage_path / f"{name}.json"
                if file_path.exists():
                    file_path.unlink()

            logger.info(f"Deleted prompt: {name}")
            return True

        return False

    def update_stats(self, name: str, success: bool, score: float = 0.0):
        """Update prompt usage statistics."""
        prompt = self.get(name)
        if not prompt:
            return

        prompt.usage_count += 1
        if success:
            prompt.success_rate = (prompt.success_rate * (prompt.usage_count - 1) + 1) / prompt.usage_count
        else:
            prompt.success_rate = (prompt.success_rate * (prompt.usage_count - 1) + 0) / prompt.usage_count

        if score > 0:
            prompt.average_score = (prompt.average_score * (prompt.usage_count - 1) + score) / prompt.usage_count

        if self.auto_save:
            self.save(prompt)


class PromptManager:
    """
    Main prompt manager with support for custom prompts, templates, and A/B testing.
    """

    def __init__(
        self,
        storage: Optional[PromptStorage] = None,
        default_prompt: str = "qa_with_context",
        enable_ab_testing: bool = False
    ):
        """
        Initialize prompt manager.

        Args:
            storage: Prompt storage instance
            default_prompt: Default prompt to use
            enable_ab_testing: Whether to enable A/B testing
        """
        self.storage = storage or PromptStorage()
        self.default_prompt = default_prompt
        self.enable_ab_testing = enable_ab_testing

        # A/B testing
        self._ab_tests: Dict[str, List[str]] = {}
        self._ab_test_weights: Dict[str, List[float]] = {}
        self._ab_test_results: Dict[str, Dict[str, Any]] = {}

        # Cache
        self._cache = CacheManager()
        self._cache_ttl = 3600

        logger.info(f"PromptManager initialized: default_prompt={default_prompt}, ab_testing={enable_ab_testing}")

    def get_prompt(
        self,
        name: Optional[str] = None,
        prompt_type: Optional[PromptType] = None,
        **kwargs
    ) -> Dict[str, str]:
        """
        Get a formatted prompt.

        Args:
            name: Prompt name
            prompt_type: Prompt type (used if name not provided)
            **kwargs: Variables for template formatting

        Returns:
            Dictionary with 'prompt' and 'system_prompt' keys
        """
        # Handle A/B testing
        if self.enable_ab_testing and name and name in self._ab_tests:
            name = self._select_ab_test_variant(name)

        # Get prompt template
        template = None
        if name:
            template = self.storage.get(name)
        elif prompt_type:
            templates = self.storage.get_by_type(prompt_type)
            if templates:
                template = templates[0]  # Use first matching template

        if not template:
            # Fallback to default
            template = self.storage.get(self.default_prompt)
            if not template:
                raise ValueError(f"No prompt found for name={name}, type={prompt_type}")

        # Format template
        try:
            prompt_text = template.format(**kwargs)
        except ValueError as e:
            logger.error(f"Failed to format prompt {template.name}: {e}")
            raise

        # Prepare result
        result = {
            "prompt": prompt_text,
            "template_name": template.name,
            "template_version": template.version
        }

        if template.system_prompt:
            result["system_prompt"] = template.system_prompt

        # Update usage stats
        self.storage.update_stats(template.name, True)

        return result

    def get_prompt_by_type(
        self,
        prompt_type: PromptType,
        **kwargs
    ) -> Dict[str, str]:
        """
        Get a prompt by type.

        Args:
            prompt_type: Prompt type
            **kwargs: Variables for template formatting

        Returns:
            Dictionary with 'prompt' and 'system_prompt' keys
        """
        templates = self.storage.get_by_type(prompt_type)
        if not templates:
            raise ValueError(f"No prompts found for type: {prompt_type}")

        # If multiple, use the one with highest success rate
        template = max(templates, key=lambda t: t.success_rate)

        return self.get_prompt(template.name, **kwargs)

    def create_prompt(
        self,
        name: str,
        template: str,
        prompt_type: PromptType,
        description: str = "",
        variables: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        tags: Optional[List[str]] = None,
        author: str = "user",
        **kwargs
    ) -> PromptTemplate:
        """
        Create a custom prompt.

        Args:
            name: Prompt name
            template: Prompt template string
            prompt_type: Prompt type
            description: Prompt description
            variables: List of variable names
            system_prompt: System prompt
            tags: List of tags
            author: Author name
            **kwargs: Additional metadata

        Returns:
            Created PromptTemplate object
        """
        # Extract variables from template if not provided
        if variables is None:
            variables = re.findall(r'\{(\w+)\}', template)

        prompt = PromptTemplate(
            name=name,
            template=template,
            prompt_type=prompt_type,
            description=description,
            variables=variables,
            system_prompt=system_prompt,
            tags=tags or [],
            author=author,
            metadata=kwargs
        )

        self.storage.save(prompt)
        logger.info(f"Created custom prompt: {name}")
        return prompt

    def update_prompt(
        self,
        name: str,
        template: Optional[str] = None,
        description: Optional[str] = None,
        system_prompt: Optional[str] = None,
        tags: Optional[List[str]] = None,
        is_active: Optional[bool] = None,
        **kwargs
    ) -> Optional[PromptTemplate]:
        """
        Update an existing prompt.

        Args:
            name: Prompt name
            template: New template string
            description: New description
            system_prompt: New system prompt
            tags: New tags
            is_active: New active status
            **kwargs: Additional metadata

        Returns:
            Updated PromptTemplate or None
        """
        prompt = self.storage.get(name)
        if not prompt:
            logger.warning(f"Prompt not found: {name}")
            return None

        if template is not None:
            prompt.template = template
            # Re-extract variables
            prompt.variables = re.findall(r'\{(\w+)\}', template)

        if description is not None:
            prompt.description = description

        if system_prompt is not None:
            prompt.system_prompt = system_prompt

        if tags is not None:
            prompt.tags = tags

        if is_active is not None:
            prompt.is_active = is_active

        # Update version
        version_parts = prompt.version.split('.')
        if version_parts:
            try:
                prompt.version = f"{version_parts[0]}.{int(version_parts[1]) + 1 if len(version_parts) > 1 else 1}.0"
            except ValueError:
                prompt.version = "1.0.0"

        prompt.updated_at = datetime.now()
        prompt.metadata.update(kwargs)

        self.storage.save(prompt)
        logger.info(f"Updated prompt: {name} (version {prompt.version})")
        return prompt

    def delete_prompt(self, name: str) -> bool:
        """Delete a prompt."""
        return self.storage.delete(name)

    def duplicate_prompt(
        self,
        name: str,
        new_name: str,
        **kwargs
    ) -> Optional[PromptTemplate]:
        """
        Duplicate a prompt.

        Args:
            name: Original prompt name
            new_name: New prompt name
            **kwargs: Override attributes

        Returns:
            New PromptTemplate or None
        """
        original = self.storage.get(name)
        if not original:
            logger.warning(f"Prompt not found: {name}")
            return None

        # Create new prompt with original attributes
        prompt = PromptTemplate(
            name=new_name,
            template=kwargs.get('template', original.template),
            prompt_type=kwargs.get('prompt_type', original.prompt_type),
            description=kwargs.get('description', original.description),
            variables=kwargs.get('variables', original.variables.copy()),
            examples=kwargs.get('examples', original.examples.copy()),
            system_prompt=kwargs.get('system_prompt', original.system_prompt),
            metadata={**original.metadata, **kwargs.get('metadata', {})},
            tags=kwargs.get('tags', original.tags.copy()),
            author=kwargs.get('author', original.author)
        )

        self.storage.save(prompt)
        logger.info(f"Duplicated prompt: {name} -> {new_name}")
        return prompt

    def create_ab_test(
        self,
        test_name: str,
        prompt_names: List[str],
        weights: Optional[List[float]] = None
    ):
        """
        Create an A/B test with multiple prompt variants.

        Args:
            test_name: Name of the test
            prompt_names: List of prompt names to test
            weights: Optional weights for each variant
        """
        # Validate prompts exist
        for name in prompt_names:
            if not self.storage.get(name):
                raise ValueError(f"Prompt not found: {name}")

        self._ab_tests[test_name] = prompt_names

        if weights:
            if len(weights) != len(prompt_names):
                raise ValueError("Weights length must match prompt names length")
            total = sum(weights)
            self._ab_test_weights[test_name] = [w / total for w in weights]
        else:
            self._ab_test_weights[test_name] = [1.0 / len(prompt_names)] * len(prompt_names)

        # Initialize results
        self._ab_test_results[test_name] = {
            "prompts": prompt_names,
            "weights": self._ab_test_weights[test_name],
            "usage": [0] * len(prompt_names),
            "success": [0] * len(prompt_names),
            "total_scores": [0.0] * len(prompt_names)
        }

        logger.info(f"Created A/B test: {test_name} with {len(prompt_names)} variants")

    def _select_ab_test_variant(self, test_name: str) -> str:
        """
        Select a variant from an A/B test.

        Args:
            test_name: Name of the test

        Returns:
            Selected prompt name
        """
        if test_name not in self._ab_tests:
            return test_name

        prompts = self._ab_tests[test_name]
        weights = self._ab_test_weights.get(test_name)

        # Update weights based on performance if enabled
        if self._ab_test_weights.get(test_name) is None:
            weights = [1.0 / len(prompts)] * len(prompts)

        # Select based on weights
        selected = random.choices(prompts, weights=weights, k=1)[0]

        # Update usage tracking
        if test_name in self._ab_test_results:
            idx = prompts.index(selected)
            self._ab_test_results[test_name]["usage"][idx] += 1

        return selected

    def record_ab_test_result(
        self,
        test_name: str,
        prompt_name: str,
        success: bool,
        score: float = 0.0
    ):
        """
        Record the result of an A/B test.

        Args:
            test_name: Name of the test
            prompt_name: The prompt variant used
            success: Whether the response was successful
            score: Quality score (0-1)
        """
        if test_name not in self._ab_test_results:
            return

        results = self._ab_test_results[test_name]
        if prompt_name not in results["prompts"]:
            return

        idx = results["prompts"].index(prompt_name)

        if success:
            results["success"][idx] += 1

        results["total_scores"][idx] += score

        # Update weights based on performance
        total_usage = sum(results["usage"])
        if total_usage > 10:  # Minimum sample size
            success_rates = []
            for i in range(len(results["prompts"])):
                if results["usage"][i] > 0:
                    rate = results["success"][i] / results["usage"][i]
                    # Add score bonus
                    avg_score = results["total_scores"][i] / results["usage"][i]
                    combined = rate * 0.7 + avg_score * 0.3
                else:
                    combined = 0.0
                success_rates.append(combined)

            # Update weights with softmax
            import math
            exp_rates = [math.exp(r * 2) for r in success_rates]  # Amplify differences
            total_exp = sum(exp_rates)
            if total_exp > 0:
                new_weights = [e / total_exp for e in exp_rates]
                self._ab_test_weights[test_name] = new_weights
                results["weights"] = new_weights

    def get_ab_test_stats(self, test_name: str) -> Dict[str, Any]:
        """
        Get statistics for an A/B test.

        Args:
            test_name: Name of the test

        Returns:
            A/B test statistics
        """
        if test_name not in self._ab_test_results:
            return {}

        results = self._ab_test_results[test_name]
        stats = {
            "test_name": test_name,
            "variants": []
        }

        for i, prompt in enumerate(results["prompts"]):
            usage = results["usage"][i]
            success = results["success"][i]
            total_score = results["total_scores"][i]
            weight = results["weights"][i] if i < len(results["weights"]) else 0

            stats["variants"].append({
                "prompt": prompt,
                "usage": usage,
                "success": success,
                "success_rate": success / usage if usage > 0 else 0,
                "avg_score": total_score / usage if usage > 0 else 0,
                "weight": weight
            })

        return stats

    def list_prompts(
        self,
        prompt_type: Optional[PromptType] = None,
        active_only: bool = True,
        tags: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """List prompts with their details."""
        prompts = self.storage.list_prompts(prompt_type, active_only, tags)
        return [p.to_dict() for p in prompts]

    def get_prompt_statistics(self) -> Dict[str, Any]:
        """Get statistics about prompts."""
        prompts = self.storage.list_prompts(active_only=False)

        stats = {
            "total_prompts": len(prompts),
            "active_prompts": sum(1 for p in prompts if p.is_active),
            "by_type": {},
            "most_used": [],
            "highest_success": [],
            "avg_success_rate": 0.0
        }

        # By type
        for prompt in prompts:
            type_key = prompt.prompt_type.value
            if type_key not in stats["by_type"]:
                stats["by_type"][type_key] = 0
            stats["by_type"][type_key] += 1

        # Most used
        sorted_by_usage = sorted(prompts, key=lambda p: p.usage_count, reverse=True)
        stats["most_used"] = [
            {"name": p.name, "usage": p.usage_count}
            for p in sorted_by_usage[:5]
        ]

        # Highest success
        sorted_by_success = sorted(prompts, key=lambda p: p.success_rate, reverse=True)
        stats["highest_success"] = [
            {"name": p.name, "success_rate": p.success_rate}
            for p in sorted_by_success[:5]
        ]

        # Average success rate
        active = [p for p in prompts if p.is_active and p.usage_count > 0]
        if active:
            stats["avg_success_rate"] = sum(p.success_rate for p in active) / len(active)

        return stats


# ============================================================
# Convenience Functions
# ============================================================

_prompt_manager: Optional[PromptManager] = None


def get_prompt_manager() -> PromptManager:
    """Get global prompt manager instance."""
    global _prompt_manager
    if _prompt_manager is None:
        _prompt_manager = PromptManager()
    return _prompt_manager


def get_prompt(
    name: Optional[str] = None,
    prompt_type: Optional[PromptType] = None,
    **kwargs
) -> Dict[str, str]:
    """
    Quick function to get a formatted prompt.

    Args:
        name: Prompt name
        prompt_type: Prompt type
        **kwargs: Variables for template formatting

    Returns:
        Dictionary with 'prompt' and 'system_prompt'
    """
    manager = get_prompt_manager()
    return manager.get_prompt(name, prompt_type, **kwargs)


def create_custom_prompt(
    name: str,
    template: str,
    prompt_type: Union[str, PromptType],
    description: str = "",
    **kwargs
) -> PromptTemplate:
    """
    Quick function to create a custom prompt.

    Args:
        name: Prompt name
        template: Prompt template
        prompt_type: Prompt type
        description: Prompt description
        **kwargs: Additional attributes

    Returns:
        Created PromptTemplate
    """
    if isinstance(prompt_type, str):
        prompt_type = PromptType(prompt_type)

    manager = get_prompt_manager()
    return manager.create_prompt(name, template, prompt_type, description, **kwargs)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    print("Testing Prompt Manager...")
    print("=" * 60)

    # Initialize manager
    manager = get_prompt_manager()

    # List prompts
    print("\n📋 Available Prompts:")
    prompts = manager.list_prompts()
    for prompt in prompts[:5]:
        print(f"  {prompt['name']} ({prompt['type']}) - v{prompt['version']}")

    # Get a prompt
    print("\n📝 Getting QA prompt:")
    prompt_result = manager.get_prompt(
        name="qa_with_context",
        context="Machine learning is a subset of AI.",
        question="What is machine learning?"
    )
    print(f"  Prompt: {prompt_result['prompt'][:100]}...")
    if prompt_result.get('system_prompt'):
        print(f"  System: {prompt_result['system_prompt']}")

    # Create custom prompt
    print("\n🔧 Creating custom prompt:")
    custom = manager.create_prompt(
        name="my_custom_qa",
        template="Answer this question: {question}\n\nContext: {context}\n\nAnswer concisely:",
        prompt_type=PromptType.CUSTOM,
        description="My custom QA prompt",
        tags=["custom", "qa"],
        author="test_user"
    )
    print(f"  Created prompt: {custom.name}")

    # Create A/B test
    print("\n🧪 Creating A/B test:")
    manager.create_ab_test(
        "test_prompts",
        ["qa_with_context", "qa_basic"],
        [0.7, 0.3]
    )
    print("  Created A/B test with 2 variants")

    # Get stats
    print("\n📊 Prompt Statistics:")
    stats = manager.get_prompt_statistics()
    print(f"  Total prompts: {stats['total_prompts']}")
    print(f"  Active prompts: {stats['active_prompts']}")
    print(f"  By type: {stats['by_type']}")

    print("\n✅ Prompt Manager ready!")

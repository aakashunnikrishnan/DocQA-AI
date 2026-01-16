"""
Prompt templates for document question answering and various LLM tasks.
Provides structured prompts, formatting utilities, and template management.
"""

from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import re
from datetime import datetime


class PromptType(Enum):
    """Types of prompts available."""
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


@dataclass
class PromptTemplate:
    """Represents a prompt template with metadata."""
    name: str
    template: str
    prompt_type: PromptType
    description: str = ""
    version: str = "1.0.0"
    variables: List[str] = field(default_factory=list)
    examples: List[Dict[str, str]] = field(default_factory=list)

    def format(self, **kwargs) -> str:
        """Format the template with variables."""
        # Validate required variables
        missing_vars = [var for var in self.variables if var not in kwargs]
        if missing_vars:
            raise ValueError(f"Missing required variables: {missing_vars}")

        # Format the template
        try:
            return self.template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Unknown variable in template: {e}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "template": self.template,
            "type": self.prompt_type.value,
            "description": self.description,
            "version": self.version,
            "variables": self.variables,
            "examples": self.examples
        }


class PromptTemplates:
    """Central repository of prompt templates."""

    # Question Answering Prompts
    QA_BASIC = PromptTemplate(
        name="qa_basic",
        template="Question: {question}\n\nAnswer:",
        prompt_type=PromptType.QA,
        description="Basic QA without context",
        variables=["question"]
    )

    QA_WITH_CONTEXT = PromptTemplate(
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
        examples=[
            {
                "context": "The company was founded in 2010 in San Francisco.",
                "question": "When was the company founded?",
                "answer": "The company was founded in 2010."
            }
        ]
    )

    QA_WITH_SOURCES = PromptTemplate(
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
    )

    # Summarization Prompts
    SUMMARIZATION_CONCISE = PromptTemplate(
        name="summarization_concise",
        template="""Summarize the following text in 2-3 sentences:

Text: {text}

Concise Summary:""",
        prompt_type=PromptType.SUMMARIZATION,
        description="Concise summarization",
        variables=["text"]
    )

    SUMMARIZATION_DETAILED = PromptTemplate(
        name="summarization_detailed",
        template="""Provide a detailed summary of the following text, including key points and main arguments.

Text: {text}

Detailed Summary:""",
        prompt_type=PromptType.SUMMARIZATION,
        description="Detailed summarization with key points",
        variables=["text"]
    )

    SUMMARIZATION_BULLET_POINTS = PromptTemplate(
        name="summarization_bullet_points",
        template="""Extract the key points from the following text as bullet points.

Text: {text}

Key Points:
-""",
        prompt_type=PromptType.SUMMARIZATION,
        description="Summarize as bullet points",
        variables=["text"]
    )

    # Conversation Prompts
    CONVERSATION_SYSTEM = PromptTemplate(
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
    )

    # Extraction Prompts
    EXTRACTION_KEY_PHRASES = PromptTemplate(
        name="extraction_key_phrases",
        template="""Extract the most important key phrases from the following text.
Return them as a comma-separated list.

Text: {text}

Key phrases:""",
        prompt_type=PromptType.EXTRACTION,
        description="Extract key phrases",
        variables=["text"]
    )

    EXTRACTION_ENTITIES = PromptTemplate(
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
    )

    EXTRACTION_DATES = PromptTemplate(
        name="extraction_dates",
        template="""Extract all dates mentioned in the following text.
Format each date as YYYY-MM-DD when possible.

Text: {text}

Dates found:""",
        prompt_type=PromptType.EXTRACTION,
        description="Extract dates",
        variables=["text"]
    )

    # Classification Prompts
    CLASSIFICATION_TOPIC = PromptTemplate(
        name="classification_topic",
        template="""Classify the following text into one of these categories:
- Technology
- Business
- Science
- Legal
- Medical
- Education
- Other

Text: {text}

Category:""",
        prompt_type=PromptType.CLASSIFICATION,
        description="Topic classification",
        variables=["text"]
    )

    CLASSIFICATION_URGENCY = PromptTemplate(
        name="classification_urgency",
        template="""Classify the urgency of the following query:
- High (requires immediate attention)
- Medium (should be addressed soon)
- Low (can be handled later)

Query: {query}

Urgency:""",
        prompt_type=PromptType.CLASSIFICATION,
        description="Urgency classification",
        variables=["query"]
    )

    # RAG-Specific Prompts
    RAG_REWRITE_QUERY = PromptTemplate(
        name="rag_rewrite_query",
        template="""Given the conversation history, rewrite the follow-up question to be self-contained and clear.

Conversation history:
{history}

Follow-up question: {question}

Rewritten question:""",
        prompt_type=PromptType.QUERY_REWRITING,
        description="Rewrite queries for better retrieval",
        variables=["history", "question"]
    )

    RAG_RESPONSE_GENERATION = PromptTemplate(
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
    )

    RAG_HALLUCINATION_CHECK = PromptTemplate(
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
    )

    RAG_RESPONSE_RERANKING = PromptTemplate(
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
    )

    # Code Generation
    CODE_GENERATION = PromptTemplate(
        name="code_generation",
        template="""Generate {language} code to solve the following problem.

Problem: {description}

Requirements:
- Include comments
- Handle edge cases
- Follow best practices

Code:""",
        prompt_type=PromptType.CODE_GENERATION,
        description="Generate code from description",
        variables=["language", "description"]
    )

    # Translation
    TRANSLATION = PromptTemplate(
        name="translation",
        template="""Translate the following text from {source_lang} to {target_lang}.

Text: {text}

Translation:""",
        prompt_type=PromptType.TRANSLATION,
        description="Translate between languages",
        variables=["source_lang", "target_lang", "text"]
    )

    # Sentiment Analysis
    SENTIMENT_ANALYSIS = PromptTemplate(
        name="sentiment_analysis",
        template="""Analyze the sentiment of the following text.
Classify as: Positive, Negative, or Neutral.

Text: {text}

Sentiment:""",
        prompt_type=PromptType.SENTIMENT_ANALYSIS,
        description="Sentiment classification",
        variables=["text"]
    )

    # All templates dictionary
    ALL_TEMPLATES = {
        "qa_basic": QA_BASIC,
        "qa_with_context": QA_WITH_CONTEXT,
        "qa_with_sources": QA_WITH_SOURCES,
        "summarization_concise": SUMMARIZATION_CONCISE,
        "summarization_detailed": SUMMARIZATION_DETAILED,
        "summarization_bullet_points": SUMMARIZATION_BULLET_POINTS,
        "conversation_system": CONVERSATION_SYSTEM,
        "extraction_key_phrases": EXTRACTION_KEY_PHRASES,
        "extraction_entities": EXTRACTION_ENTITIES,
        "extraction_dates": EXTRACTION_DATES,
        "classification_topic": CLASSIFICATION_TOPIC,
        "classification_urgency": CLASSIFICATION_URGENCY,
        "rag_rewrite_query": RAG_REWRITE_QUERY,
        "rag_response_generation": RAG_RESPONSE_GENERATION,
        "rag_hallucination_check": RAG_HALLUCINATION_CHECK,
        "rag_response_reranking": RAG_RESPONSE_RERANKING,
        "code_generation": CODE_GENERATION,
        "translation": TRANSLATION,
        "sentiment_analysis": SENTIMENT_ANALYSIS
    }

    @classmethod
    def get_template(cls, name: str) -> PromptTemplate:
        """Get a template by name."""
        if name not in cls.ALL_TEMPLATES:
            raise KeyError(f"Template '{name}' not found. Available: {list(cls.ALL_TEMPLATES.keys())}")
        return cls.ALL_TEMPLATES[name]

    @classmethod
    def list_templates(cls, prompt_type: Optional[PromptType] = None) -> List[Dict[str, Any]]:
        """List all templates, optionally filtered by type."""
        templates = cls.ALL_TEMPLATES.values()
        if prompt_type:
            templates = [t for t in templates if t.prompt_type == prompt_type]
        return [t.to_dict() for t in templates]


class PromptFormatter:
    """Utility for formatting prompts with context and history."""

    @staticmethod
    def format_context(chunks: List[Dict[str, Any]], max_chars: int = 4000) -> str:
        """
        Format retrieved chunks into a single context string.

        Args:
            chunks: List of chunks with 'text' and optional 'source'
            max_chars: Maximum characters for context

        Returns:
            Formatted context string
        """
        context_parts = []
        total_chars = 0

        for i, chunk in enumerate(chunks, 1):
            text = chunk.get('text', '')
            source = chunk.get('source', chunk.get('document_name', f'Document {i}'))

            formatted = f"[Source: {source}]\n{text}\n"

            if total_chars + len(formatted) > max_chars:
                # Truncate last chunk if needed
                remaining = max_chars - total_chars
                if remaining > 100:
                    formatted = formatted[:remaining] + "..."
                else:
                    break

            context_parts.append(formatted)
            total_chars += len(formatted)

        return "\n".join(context_parts)

    @staticmethod
    def format_conversation_history(
        history: List[Dict[str, str]],
        max_turns: int = 5,
        max_tokens_per_turn: int = 500
    ) -> str:
        """
        Format conversation history for prompt inclusion.

        Args:
            history: List of {'role': 'user'/'assistant', 'content': str}
            max_turns: Maximum number of conversation turns to include
            max_tokens_per_turn: Maximum characters per turn

        Returns:
            Formatted conversation history
        """
        if not history:
            return "No previous conversation."

        # Take only recent turns
        recent_history = history[-max_turns * 2:]  # Each turn has user + assistant

        formatted = []
        for turn in recent_history:
            role = turn.get('role', 'unknown')
            content = turn.get('content', '')

            # Truncate if too long
            if len(content) > max_tokens_per_turn:
                content = content[:max_tokens_per_turn] + "..."

            if role == 'user':
                formatted.append(f"User: {content}")
            elif role == 'assistant':
                formatted.append(f"Assistant: {content}")

        return "\n".join(formatted)

    @staticmethod
    def format_chunks_with_scores(
        chunks: List[Dict[str, Any]],
        include_scores: bool = True,
        include_metadata: bool = False
    ) -> str:
        """
        Format chunks with relevance scores.

        Args:
            chunks: List of chunks with 'text', 'score', and optional 'metadata'
            include_scores: Whether to include relevance scores
            include_metadata: Whether to include metadata

        Returns:
            Formatted chunks string
        """
        formatted_parts = []

        for i, chunk in enumerate(chunks, 1):
            parts = []

            # Header with score
            header = f"Document {i}"
            if include_scores and 'score' in chunk:
                score = chunk['score']
                header += f" (Relevance: {score:.3f})"
            parts.append(header)

            # Metadata
            if include_metadata and 'metadata' in chunk:
                metadata = chunk['metadata']
                if metadata:
                    parts.append(f"Metadata: {json.dumps(metadata, indent=2)}")

            # Content
            parts.append(chunk.get('text', ''))

            formatted_parts.append("\n".join(parts))

        return "\n\n---\n\n".join(formatted_parts)


class PromptOptimizer:
    """Optimize prompts for better performance."""

    @staticmethod
    def truncate_prompt(prompt: str, max_tokens: int, tokenizer=None) -> str:
        """
        Truncate prompt to fit within token limit.

        Args:
            prompt: The prompt text
            max_tokens: Maximum allowed tokens
            tokenizer: Tokenizer function (if None, uses rough estimate)

        Returns:
            Truncated prompt
        """
        if tokenizer:
            tokens = tokenizer(prompt)
            if len(tokens) <= max_tokens:
                return prompt

            # Truncate by characters (rough approximation)
            ratio = max_tokens / len(tokens)
            new_length = int(len(prompt) * ratio * 0.9)  # Conservative
            return prompt[:new_length] + "\n...[truncated]"
        else:
            # Rough estimate: 1 token ≈ 4 characters
            max_chars = max_tokens * 4
            if len(prompt) <= max_chars:
                return prompt
            return prompt[:max_chars] + "\n...[truncated]"

    @staticmethod
    def add_instruction_prefix(prompt: str, instruction: str = "Please follow these instructions carefully:") -> str:
        """Add instruction prefix to prompt."""
        return f"{instruction}\n\n{prompt}"

    @staticmethod
    def add_examples(prompt: str, examples: List[Dict[str, str]], example_template: str = "Input: {input}\nOutput: {output}") -> str:
        """Add few-shot examples to prompt."""
        if not examples:
            return prompt

        examples_text = "Examples:\n\n"
        for ex in examples:
            examples_text += example_template.format(**ex) + "\n\n"

        return examples_text + prompt

    @staticmethod
    def add_system_prompt(prompt: str, system_prompt: str) -> str:
        """Add system prompt to user prompt."""
        return f"{system_prompt}\n\n{prompt}"


class PromptBuilder:
    """Builder class for constructing complex prompts."""

    def __init__(self):
        self.components = []

    def add_text(self, text: str) -> 'PromptBuilder':
        """Add raw text."""
        if text:
            self.components.append(text)
        return self

    def add_context(self, context: str, label: str = "Context") -> 'PromptBuilder':
        """Add context section."""
        if context:
            self.components.append(f"{label}:\n{context}")
        return self

    def add_question(self, question: str) -> 'PromptBuilder':
        """Add question section."""
        self.components.append(f"Question: {question}")
        return self

    def add_instructions(self, instructions: List[str]) -> 'PromptBuilder':
        """Add instructions list."""
        if instructions:
            instructions_text = "Instructions:\n" + "\n".join(f"- {inst}" for inst in instructions)
            self.components.append(instructions_text)
        return self

    def add_examples(self, examples: List[Dict[str, str]]) -> 'PromptBuilder':
        """Add examples."""
        if examples:
            examples_text = "Examples:\n"
            for ex in examples:
                examples_text += f"Input: {ex.get('input', '')}\nOutput: {ex.get('output', '')}\n\n"
            self.components.append(examples_text)
        return self

    def add_format_requirements(self, format_type: str = "concise") -> 'PromptBuilder':
        """Add output format requirements."""
        formats = {
            "concise": "Provide a concise answer.",
            "detailed": "Provide a detailed answer with explanation.",
            "bullet_points": "Provide answer as bullet points.",
            "json": "Provide answer in JSON format.",
            "numbered": "Provide answer as numbered list."
        }
        self.components.append(f"Format: {formats.get(format_type, format_type)}")
        return self

    def add_constraints(self, constraints: List[str]) -> 'PromptBuilder':
        """Add constraints."""
        if constraints:
            constraints_text = "Constraints:\n" + "\n".join(f"- {c}" for c in constraints)
            self.components.append(constraints_text)
        return self

    def build(self, separator: str = "\n\n") -> str:
        """Build the final prompt."""
        return separator.join(self.components)

    def build_with_answer_suffix(self) -> str:
        """Build prompt with answer suffix."""
        prompt = self.build()
        return f"{prompt}\n\nAnswer:"


# Convenience functions
def get_qa_prompt(
    question: str,
    context: Optional[str] = None,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    include_sources: bool = False
) -> str:
    """
    Get a formatted QA prompt.

    Args:
        question: User question
        context: Retrieved context
        conversation_history: Previous conversation turns
        include_sources: Whether to include source attribution

    Returns:
        Formatted prompt
    """
    if context:
        if include_sources:
            template = PromptTemplates.get_template("qa_with_sources")
            return template.format(question=question, sources=context)
        else:
            template = PromptTemplates.get_template("qa_with_context")
            return template.format(question=question, context=context)
    else:
        template = PromptTemplates.get_template("qa_basic")
        return template.format(question=question)


def get_rag_prompt(
    question: str,
    chunks: List[Dict[str, Any]],
    history: Optional[List[Dict[str, str]]] = None,
    max_context_chars: int = 4000
) -> str:
    """
    Get a complete RAG prompt with formatted context.

    Args:
        question: User question
        chunks: Retrieved chunks with text and scores
        history: Conversation history
        max_context_chars: Maximum context length

    Returns:
        Complete RAG prompt
    """
    # Format context
    context = PromptFormatter.format_context(chunks, max_chars=max_context_chars)

    # Format history
    history_text = ""
    if history:
        history_text = PromptFormatter.format_conversation_history(history)

    if history_text:
        template = PromptTemplates.get_template("conversation_system")
        return template.format(
            context=context,
            history=history_text,
            question=question
        )
    else:
        template = PromptTemplates.get_template("rag_response_generation")
        return template.format(
            context=context,
            question=question
        )


if __name__ == "__main__":
    # Example usage
    print("Available templates:")
    for template in PromptTemplates.list_templates()[:5]:
        print(f"  - {template['name']}: {template['description']}")

    # Use a template
    template = PromptTemplates.get_template("qa_with_context")
    prompt = template.format(
        context="The sky is blue because of Rayleigh scattering.",
        question="Why is the sky blue?"
    )

    print("\nGenerated prompt:")
    print("-" * 50)
    print(prompt)
    print("-" * 50)

    # Use prompt builder
    prompt = (PromptBuilder()
              .add_instructions(["Answer based on context", "Be concise"])
              .add_context("The company has 500 employees worldwide.")
              .add_question("How many employees does the company have?")
              .add_format_requirements("concise")
              .build_with_answer_suffix())

    print("\nBuilder example:")
    print("-" * 50)
    print(prompt)
    print("-" * 50)

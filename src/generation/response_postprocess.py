"""
Response post-processing module for cleaning, formatting, and validating LLM responses.
Handles response parsing, confidence scoring, and quality checks.
"""

import re
import json
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from string import punctuation

logger = logging.getLogger(__name__)


class ResponseFormat(Enum):
    """Supported response formats."""
    TEXT = "text"
    JSON = "json"
    BULLET_POINTS = "bullet_points"
    NUMBERED_LIST = "numbered_list"
    CODE = "code"
    TABLE = "table"


@dataclass
class ProcessedResponse:
    """Processed response with metadata."""
    original_text: str
    cleaned_text: str
    format: ResponseFormat
    confidence: float = 1.0
    tokens_used: int = 0
    has_hallucination: bool = False
    hallucination_score: float = 0.0
    sources: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "original_text": self.original_text,
            "cleaned_text": self.cleaned_text,
            "format": self.format.value,
            "confidence": self.confidence,
            "has_hallucination": self.has_hallucination,
            "hallucination_score": self.hallucination_score,
            "sources": self.sources,
            "metadata": self.metadata
        }

    def get_final_response(self) -> str:
        """Get the final response text to display."""
        if self.has_hallucination and self.hallucination_score > 0.5:
            return f"[Warning: Response may contain unsupported information]\n\n{self.cleaned_text}"
        return self.cleaned_text


class ResponseCleaner:
    """Clean and normalize response text."""

    @staticmethod
    def remove_think_tags(text: str) -> str:
        """Remove  tags and their content."""
        # Remove  tags (common in some LLMs)
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        text = re.sub(r'\[THINK\].*?\[/THINK\]', '', text, flags=re.DOTALL)
        return text.strip()

    @staticmethod
    def remove_extra_whitespace(text: str) -> str:
        """Remove extra whitespace and normalize spacing."""
        # Replace multiple newlines with double newline
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        # Replace multiple spaces with single space
        text = re.sub(r' +', ' ', text)
        # Remove spaces before punctuation
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        return text.strip()

    @staticmethod
    def remove_markdown_formatting(text: str) -> str:
        """Remove common markdown formatting."""
        # Remove bold/italic
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'\*([^*]+)\*', r'\1', text)
        text = re.sub(r'__([^_]+)__', r'\1', text)
        text = re.sub(r'_([^_]+)_', r'\1', text)

        # Remove links
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)

        # Remove code blocks
        text = re.sub(r'```[\s\S]*?```', '', text)
        text = re.sub(r'`([^`]+)`', r'\1', text)

        return text.strip()

    @staticmethod
    def remove_repetitions(text: str) -> str:
        """Remove repeated phrases and words."""
        # Remove repeated words (e.g., "the the" -> "the")
        text = re.sub(r'\b(\w+)(\s+\1\b)+', r'\1', text, flags=re.IGNORECASE)

        # Remove repeated punctuation
        text = re.sub(r'([!?.]){2,}', r'\1', text)

        return text

    @staticmethod
    def fix_capitalization(text: str) -> str:
        """Fix capitalization issues."""
        # Capitalize first letter of each sentence
        sentences = re.split(r'([.!?] +)', text)
        for i in range(0, len(sentences), 2):
            if sentences[i]:
                sentences[i] = sentences[i][0].upper() + sentences[i][1:] if sentences[i] else sentences[i]

        return ''.join(sentences)

    @staticmethod
    def remove_incomplete_sentences(text: str) -> str:
        """Remove incomplete sentences at the end."""
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)

        # Check if last sentence is complete (ends with punctuation)
        if sentences and not re.search(r'[.!?]$', sentences[-1]):
            sentences = sentences[:-1]

        return ' '.join(sentences)

    @classmethod
    def clean(cls, text: str, aggressive: bool = False) -> str:
        """
        Clean response text.

        Args:
            text: Raw response text
            aggressive: Whether to apply aggressive cleaning

        Returns:
            Cleaned text
        """
        if not text:
            return ""

        # Basic cleaning
        text = cls.remove_think_tags(text)
        text = cls.remove_extra_whitespace(text)

        if aggressive:
            text = cls.remove_markdown_formatting(text)
            text = cls.remove_repetitions(text)
            text = cls.fix_capitalization(text)
            text = cls.remove_incomplete_sentences(text)

        return text


class ResponseParser:
    """Parse responses into structured formats."""

    @staticmethod
    def parse_json(text: str) -> Optional[Dict[str, Any]]:
        """
        Parse JSON from response text.

        Args:
            text: Response text containing JSON

        Returns:
            Parsed JSON dict or None if parsing fails
        """
        # Try to extract JSON from code blocks
        json_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', text)
        if json_match:
            text = json_match.group(1)

        # Try to find JSON object
        json_match = re.search(r'(\{[\s\S]*\})', text)
        if json_match:
            text = json_match.group(1)

        # Parse JSON
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON response")
            return None

    @staticmethod
    def parse_bullet_points(text: str) -> List[str]:
        """
        Parse bullet points from response.

        Args:
            text: Response text with bullet points

        Returns:
            List of bullet point items
        """
        bullet_points = []

        # Common bullet point patterns
        patterns = [
            r'^[\s]*[-*•]\s+(.+)$',  # -, *, •
            r'^[\s]*\d+[\.\)]\s+(.+)$',  # 1., 2), etc.
            r'^[\s]*[a-z][\.\)]\s+(.+)$',  # a., b), etc.
        ]

        lines = text.split('\n')
        for line in lines:
            for pattern in patterns:
                match = re.match(pattern, line.strip())
                if match:
                    bullet_points.append(match.group(1).strip())
                    break

        return bullet_points

    @staticmethod
    def parse_numbered_list(text: str) -> List[str]:
        """
        Parse numbered list from response.

        Args:
            text: Response text with numbered list

        Returns:
            List of numbered items
        """
        items = []
        pattern = r'^\s*\d+[\.\)]\s+(.+)$'

        lines = text.split('\n')
        for line in lines:
            match = re.match(pattern, line.strip())
            if match:
                items.append(match.group(1).strip())

        return items

    @staticmethod
    def parse_code_blocks(text: str, language: Optional[str] = None) -> List[str]:
        """
        Extract code blocks from response.

        Args:
            text: Response text with code blocks
            language: Optional language filter

        Returns:
            List of code block contents
        """
        if language:
            pattern = rf'```{language}\s*([\s\S]*?)```'
        else:
            pattern = r'```(?:\w+)?\s*([\s\S]*?)```'

        matches = re.findall(pattern, text, re.MULTILINE)
        return [match.strip() for match in matches]

    @staticmethod
    def parse_table(text: str) -> List[List[str]]:
        """
        Parse markdown table from response.

        Args:
            text: Response text with markdown table

        Returns:
            2D list of table cells
        """
        lines = text.split('\n')
        table_lines = []
        in_table = False

        for line in lines:
            if '|' in line and line.strip().startswith('|'):
                in_table = True
                table_lines.append(line)
            elif in_table and not line.strip():
                break

        if not table_lines:
            return []

        # Parse table
        table = []
        for i, line in enumerate(table_lines):
            # Skip separator line (|---|)
            if re.match(r'[\s\|:\-]+$', line):
                continue

            cells = [cell.strip() for cell in line.split('|')[1:-1]]
            table.append(cells)

        return table


class ConfidenceScorer:
    """Score response confidence based on various signals."""

    def __init__(self):
        self.low_confidence_phrases = [
            "i think", "i believe", "maybe", "perhaps", "possibly",
            "not sure", "uncertain", "unclear", "i don't know",
            "it seems", "apparently", "probably"
        ]

        self.hedging_phrases = [
            "could be", "might be", "may be", "likely",
            "based on my understanding", "as far as i know"
        ]

        self.uncertainty_markers = [
            "?",
            "not certain",
            "not confident",
            "guess"
        ]

    def score(self, response: str, context_available: bool = True) -> float:
        """
        Score confidence of a response.

        Args:
            response: Response text
            context_available: Whether context was available

        Returns:
            Confidence score between 0 and 1
        """
        score = 1.0

        # Check for low confidence phrases
        response_lower = response.lower()
        for phrase in self.low_confidence_phrases:
            if phrase in response_lower:
                score -= 0.15
                break

        # Check for hedging
        for phrase in self.hedging_phrases:
            if phrase in response_lower:
                score -= 0.1
                break

        # Check for uncertainty markers
        for marker in self.uncertainty_markers:
            if marker in response_lower:
                score -= 0.05

        # Check response length (very short responses may be low confidence)
        if len(response.split()) < 5:
            score -= 0.2

        # Boost if response ends with period (complete thought)
        if response.strip().endswith('.'):
            score += 0.05

        # Penalize if response contains "I cannot answer" or similar
        if any(phrase in response_lower for phrase in ["cannot answer", "unable to answer", "no information"]):
            score -= 0.3

        # Context availability impacts confidence
        if not context_available:
            score -= 0.2

        # Clamp to [0, 1]
        return max(0.0, min(1.0, score))


class HallucinationDetector:
    """Detect potential hallucinations in responses."""

    def __init__(self):
        self.hallucination_patterns = [
            # Specific numbers without source
            r'\b\d+%\b',
            r'\$\d+(?:,\d+)*',
            # Absolute statements
            r'\b(always|never|everyone|no one|absolutely|certainly)\b',
            # Specific dates
            r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b',
        ]

        self.uncertainty_words = [
            'maybe', 'perhaps', 'possibly', 'likely', 'unlikely',
            'could', 'might', 'may', 'seems', 'appears'
        ]

    def detect(self, response: str, context: Optional[str] = None) -> Tuple[bool, float]:
        """
        Detect potential hallucinations.

        Args:
            response: Response text
            context: Optional context for comparison

        Returns:
            Tuple of (has_hallucination, hallucination_score)
        """
        score = 0.0
        response_lower = response.lower()

        # Check for hallucination patterns
        for pattern in self.hallucination_patterns:
            if re.search(pattern, response):
                score += 0.1

        # Check for lack of uncertainty words in speculative content
        if score > 0.2 and not any(word in response_lower for word in self.uncertainty_words):
            score += 0.15

        # Check response length vs patterns
        if len(response.split()) > 100 and score > 0.3:
            score += 0.1

        # Check for contradictory statements within response
        if self._has_contradictions(response):
            score += 0.25

        # Clamp score
        score = min(1.0, score)
        has_hallucination = score > 0.3

        return has_hallucination, score

    def _has_contradictions(self, text: str) -> bool:
        """Check for contradictory statements in text."""
        # Simple contradiction detection
        contradiction_pairs = [
            (r'\byes\b', r'\bno\b'),
            (r'\btrue\b', r'\bfalse\b'),
            (r'\balways\b', r'\bnever\b'),
            (r'\ball\b', r'\bnone\b'),
        ]

        for pattern1, pattern2 in contradiction_pairs:
            if re.search(pattern1, text.lower()) and re.search(pattern2, text.lower()):
                return True

        return False


class SourceExtractor:
    """Extract and validate source citations from responses."""

    @staticmethod
    def extract_sources(text: str) -> List[str]:
        """
        Extract source citations from response.

        Args:
            text: Response text

        Returns:
            List of source identifiers
        """
        sources = []

        # Pattern for source citations
        patterns = [
            r'\[Source:\s*([^\]]+)\]',
            r'\(Source:\s*([^\)]+)\)',
            r'from\s+([^,.]+?)(?:\s+document|\s+page|\s+section|\.|$)',
            r'according to (?:the )?([^,.]+)',
            r'cited from ([^,.]+)',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            sources.extend(matches)

        # Remove duplicates and clean
        sources = list(set([s.strip() for s in sources]))

        return sources

    @staticmethod
    def validate_sources(sources: List[str], available_sources: List[str]) -> List[str]:
        """
        Validate that cited sources exist in available sources.

        Args:
            sources: Cited sources from response
            available_sources: Available source identifiers

        Returns:
            List of valid sources
        """
        valid_sources = []

        for source in sources:
            source_lower = source.lower()
            for available in available_sources:
                if source_lower in available.lower() or available.lower() in source_lower:
                    valid_sources.append(source)
                    break

        return valid_sources


class ResponsePostProcessor:
    """Main post-processing pipeline for LLM responses."""

    def __init__(
        self,
        aggressive_cleaning: bool = False,
        enable_confidence_scoring: bool = True,
        enable_hallucination_detection: bool = True,
        enable_source_extraction: bool = True
    ):
        """
        Initialize response post-processor.

        Args:
            aggressive_cleaning: Apply aggressive cleaning
            enable_confidence_scoring: Enable confidence scoring
            enable_hallucination_detection: Enable hallucination detection
            enable_source_extraction: Enable source extraction
        """
        self.aggressive_cleaning = aggressive_cleaning
        self.enable_confidence_scoring = enable_confidence_scoring
        self.enable_hallucination_detection = enable_hallucination_detection
        self.enable_source_extraction = enable_source_extraction

        self.cleaner = ResponseCleaner()
        self.parser = ResponseParser()
        self.confidence_scorer = ConfidenceScorer()
        self.hallucination_detector = HallucinationDetector()
        self.source_extractor = SourceExtractor()

    def process(
        self,
        response: str,
        context: Optional[str] = None,
        available_sources: Optional[List[str]] = None,
        expected_format: Optional[ResponseFormat] = None
    ) -> ProcessedResponse:
        """
        Process a raw LLM response.

        Args:
            response: Raw response text
            context: Optional context for hallucination detection
            available_sources: Available source identifiers
            expected_format: Expected response format

        Returns:
            ProcessedResponse object
        """
        if not response:
            return ProcessedResponse(
                original_text="",
                cleaned_text="",
                format=expected_format or ResponseFormat.TEXT,
                confidence=0.0
            )

        original_text = response

        # Clean response
        cleaned_text = self.cleaner.clean(response, aggressive=self.aggressive_cleaning)

        # Detect format
        detected_format = self._detect_format(cleaned_text, expected_format)

        # Score confidence
        confidence = 1.0
        if self.enable_confidence_scoring:
            confidence = self.confidence_scorer.score(cleaned_text, context is not None)

        # Detect hallucinations
        has_hallucination = False
        hallucination_score = 0.0
        if self.enable_hallucination_detection:
            has_hallucination, hallucination_score = self.hallucination_detector.detect(
                cleaned_text, context
            )

        # Extract sources
        sources = []
        if self.enable_source_extraction:
            sources = self.source_extractor.extract_sources(cleaned_text)
            if available_sources:
                sources = self.source_extractor.validate_sources(sources, available_sources)

        return ProcessedResponse(
            original_text=original_text,
            cleaned_text=cleaned_text,
            format=detected_format,
            confidence=confidence,
            has_hallucination=has_hallucination,
            hallucination_score=hallucination_score,
            sources=sources
        )

    def _detect_format(
        self,
        text: str,
        expected_format: Optional[ResponseFormat]
    ) -> ResponseFormat:
        """Detect response format."""
        if expected_format:
            return expected_format

        # Auto-detect format
        if self.parser.parse_json(text):
            return ResponseFormat.JSON

        if self.parser.parse_bullet_points(text):
            return ResponseFormat.BULLET_POINTS

        if self.parser.parse_numbered_list(text):
            return ResponseFormat.NUMBERED_LIST

        if self.parser.parse_code_blocks(text):
            return ResponseFormat.CODE

        if self.parser.parse_table(text):
            return ResponseFormat.TABLE

        return ResponseFormat.TEXT

    def process_batch(
        self,
        responses: List[str],
        contexts: Optional[List[str]] = None,
        available_sources: Optional[List[str]] = None
    ) -> List[ProcessedResponse]:
        """
        Process multiple responses.

        Args:
            responses: List of raw responses
            contexts: Optional list of contexts
            available_sources: Available sources

        Returns:
            List of ProcessedResponse objects
        """
        processed = []

        for i, response in enumerate(responses):
            context = contexts[i] if contexts and i < len(contexts) else None
            processed.append(self.process(response, context, available_sources))

        return processed


# Convenience functions
def postprocess_response(
    response: str,
    context: Optional[str] = None,
    aggressive_cleaning: bool = False
) -> ProcessedResponse:
    """
    Quick post-processing of a single response.

    Args:
        response: Raw response text
        context: Optional context
        aggressive_cleaning: Apply aggressive cleaning

    Returns:
        ProcessedResponse object
    """
    processor = ResponsePostProcessor(aggressive_cleaning=aggressive_cleaning)
    return processor.process(response, context)


def clean_response(response: str, aggressive: bool = False) -> str:
    """
    Quick cleaning of response text.

    Args:
        response: Raw response text
        aggressive: Apply aggressive cleaning

    Returns:
        Cleaned response text
    """
    cleaner = ResponseCleaner()
    cleaned = cleaner.clean(response, aggressive=aggressive)
    return cleaned


if __name__ == "__main__":
    # Example usage
    processor = ResponsePostProcessor(aggressive_cleaning=True)

    # Test responses
    test_responses = [
        "The answer is 42. I think that's correct.",
        """
        Based on the context, the company was founded in 2010 in San Francisco.
        
        Here are the key points:
        - Founded in 2010
        - Location: San Francisco
        - Initial funding: $1M
        """,
        """
        <think>Let me analyze this carefully...</think>
        The sky appears blue due to Rayleigh scattering.
        """,
        "I cannot answer this question based on the provided documents.",
    ]

    for response in test_responses:
        processed = processor.process(response)
        print(f"\nOriginal: {response[:100]}...")
        print(f"Cleaned: {processed.cleaned_text[:100]}...")
        print(f"Confidence: {processed.confidence:.2f}")
        print(f"Has hallucination: {processed.has_hallucination}")
        if processed.sources:
            print(f"Sources: {processed.sources}")
        print("-" * 50)

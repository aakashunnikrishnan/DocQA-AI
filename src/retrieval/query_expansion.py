"""
Query expansion module for improving retrieval quality.
Supports multiple expansion strategies:
- Synonym expansion (WordNet, embeddings)
- Context-aware expansion
- Query decomposition
- Sparse expansion (TF-IDF)
- Dense expansion (embeddings)
- FAQ-style expansion
"""

import re
import logging
from typing import List, Dict, Any, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import math
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Try importing NLP libraries
try:
    from nltk.corpus import wordnet
    from nltk.stem import WordNetLemmatizer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logger.warning("nltk not installed. Install with: pip install nltk")

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logger.warning("spacy not installed. Install with: pip install spacy")

try:
    from sentence_transformers import SentenceTransformer, util
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers not installed")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not installed")


class ExpansionStrategy(Enum):
    """Query expansion strategies."""
    SYNONYM = "synonym"
    EMBEDDING = "embedding"
    CONTEXT = "context"
    DECOMPOSITION = "decomposition"
    HYBRID = "hybrid"
    FAQ = "faq"
    SPARSE = "sparse"
    DENSE = "dense"


class ExpansionMode(Enum):
    """Modes for combining expanded terms."""
    BOOST = "boost"  # Boost original terms
    REPLACE = "replace"  # Replace with expanded terms
    EXTEND = "extend"  # Add expanded terms
    WEIGHTED = "weighted"  # Weighted combination


@dataclass
class ExpandedTerm:
    """Represents an expanded term with metadata."""
    term: str
    original_term: str
    weight: float = 1.0
    source: str = ""
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "term": self.term,
            "original_term": self.original_term,
            "weight": self.weight,
            "source": self.source,
            "confidence": self.confidence
        }


@dataclass
class ExpansionResult:
    """Result of query expansion."""
    original_query: str
    expanded_query: str
    expanded_terms: List[ExpandedTerm]
    strategy: ExpansionStrategy
    mode: ExpansionMode
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "original_query": self.original_query,
            "expanded_query": self.expanded_query,
            "expanded_terms": [t.to_dict() for t in self.expanded_terms],
            "strategy": self.strategy.value,
            "mode": self.mode.value,
            "confidence": self.confidence,
            "metadata": self.metadata
        }


class Tokenizer:
    """Simple tokenizer for query expansion."""

    def __init__(self, language: str = "en"):
        self.language = language

        # Common stopwords
        self.stopwords = {
            'a', 'an', 'the', 'of', 'to', 'for', 'with', 'on', 'at', 'from',
            'by', 'in', 'as', 'is', 'was', 'were', 'are', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'must'
        }

        # Punctuation
        self.punctuation = r'[^\w\s]'

    def tokenize(self, text: str, remove_stopwords: bool = True) -> List[str]:
        """
        Tokenize text.

        Args:
            text: Text to tokenize
            remove_stopwords: Whether to remove stopwords

        Returns:
            List of tokens
        """
        # Convert to lowercase
        text = text.lower()

        # Remove punctuation
        text = re.sub(self.punctuation, ' ', text)

        # Split into tokens
        tokens = text.split()

        # Remove stopwords if requested
        if remove_stopwords:
            tokens = [t for t in tokens if t not in self.stopwords]

        return tokens

    def detokenize(self, tokens: List[str]) -> str:
        """Convert tokens back to text."""
        return ' '.join(tokens)


class SynonymExpander:
    """
    Query expansion using synonyms from WordNet and embeddings.
    """

    def __init__(
        self,
        max_synonyms: int = 5,
        similarity_threshold: float = 0.5,
        use_wordnet: bool = True,
        use_embeddings: bool = True,
        embedding_model: Optional[str] = "all-MiniLM-L6-v2"
    ):
        """
        Initialize synonym expander.

        Args:
            max_synonyms: Maximum synonyms per term
            similarity_threshold: Minimum similarity for embeddings
            use_wordnet: Whether to use WordNet
            use_embeddings: Whether to use embeddings
            embedding_model: Sentence transformer model
        """
        self.max_synonyms = max_synonyms
        self.similarity_threshold = similarity_threshold
        self.use_wordnet = use_wordnet and NLTK_AVAILABLE
        self.use_embeddings = use_embeddings and SENTENCE_TRANSFORMERS_AVAILABLE

        # Initialize NLTK
        if self.use_wordnet:
            try:
                self.lemmatizer = WordNetLemmatizer()
                # Download required NLTK data
                import nltk
                try:
                    nltk.data.find('corpora/wordnet')
                except LookupError:
                    nltk.download('wordnet', quiet=True)
            except Exception as e:
                logger.warning(f"Failed to initialize WordNet: {e}")
                self.use_wordnet = False

        # Initialize embedding model
        if self.use_embeddings:
            try:
                self.embedding_model = SentenceTransformer(embedding_model)
                logger.info(f"Initialized embedding model: {embedding_model}")
            except Exception as e:
                logger.warning(f"Failed to initialize embedding model: {e}")
                self.use_embeddings = False

        self.tokenizer = Tokenizer()

        logger.info(f"SynonymExpander initialized: max_synonyms={max_synonyms}, "
                   f"use_wordnet={use_wordnet}, use_embeddings={use_embeddings}")

    def expand_term(self, term: str) -> List[ExpandedTerm]:
        """
        Expand a single term with synonyms.

        Args:
            term: Term to expand

        Returns:
            List of ExpandedTerm objects
        """
        expanded = []

        # WordNet synonyms
        if self.use_wordnet:
            wordnet_synonyms = self._get_wordnet_synonyms(term)
            for synonym in wordnet_synonyms[:self.max_synonyms]:
                expanded.append(ExpandedTerm(
                    term=synonym,
                    original_term=term,
                    weight=0.8,
                    source="wordnet",
                    confidence=0.8
                ))

        # Embedding-based synonyms
        if self.use_embeddings:
            embedding_synonyms = self._get_embedding_synonyms(term)
            for synonym in embedding_synonyms[:self.max_synonyms]:
                expanded.append(ExpandedTerm(
                    term=synonym,
                    original_term=term,
                    weight=0.7,
                    source="embedding",
                    confidence=0.7
                ))

        return expanded

    def _get_wordnet_synonyms(self, term: str) -> List[str]:
        """Get WordNet synonyms for a term."""
        synonyms = set()

        try:
            # Get synsets
            synsets = wordnet.synsets(term)

            for synset in synsets:
                # Get lemmas
                for lemma in synset.lemmas():
                    synonym = lemma.name().replace('_', ' ')
                    if synonym.lower() != term.lower():
                        synonyms.add(synonym)

        except Exception as e:
            logger.debug(f"WordNet error for term '{term}': {e}")

        return list(synonyms)

    def _get_embedding_synonyms(self, term: str) -> List[str]:
        """Get embedding-based synonyms."""
        if not self.use_embeddings or not hasattr(self, 'embedding_model'):
            return []

        try:
            # TODO: For production, maintain a vocabulary of terms
            # For now, return empty list
            return []

        except Exception as e:
            logger.debug(f"Embedding synonym error for term '{term}': {e}")
            return []

    def expand_query(self, query: str, mode: ExpansionMode = ExpansionMode.EXTEND) -> ExpansionResult:
        """
        Expand a query with synonyms.

        Args:
            query: Query string
            mode: Expansion mode

        Returns:
            ExpansionResult object
        """
        tokens = self.tokenizer.tokenize(query, remove_stopwords=False)

        all_expanded_terms = []
        expanded_tokens = []
        seen_terms = set()

        for token in tokens:
            # Add original term
            if token not in seen_terms:
                expanded_tokens.append(token)
                seen_terms.add(token)

            # Get synonyms
            synonyms = self.expand_term(token)
            all_expanded_terms.extend(synonyms)

            # Add synonyms based on mode
            if mode == ExpansionMode.EXTEND or mode == ExpansionMode.BOOST:
                for synonym in synonyms[:2]:  # Limit to top synonyms
                    if synonym.term not in seen_terms:
                        expanded_tokens.append(synonym.term)
                        seen_terms.add(synonym.term)
            elif mode == ExpansionMode.REPLACE:
                # Replace original with synonyms
                expanded_tokens.remove(token)
                for synonym in synonyms[:2]:
                    expanded_tokens.append(synonym.term)
                    seen_terms.add(synonym.term)

        # Build expanded query
        expanded_query = self.tokenizer.detokenize(expanded_tokens)

        # Calculate confidence
        confidence = min(1.0, len(all_expanded_terms) / (len(tokens) + 1) * 2)

        return ExpansionResult(
            original_query=query,
            expanded_query=expanded_query,
            expanded_terms=all_expanded_terms,
            strategy=ExpansionStrategy.SYNONYM,
            mode=mode,
            confidence=confidence,
            metadata={
                "original_tokens": tokens,
                "expanded_tokens": expanded_tokens,
                "num_synonyms": len(all_expanded_terms)
            }
        )


class ContextualExpander:
    """
    Context-aware query expansion using surrounding text or conversation history.
    """

    def __init__(
        self,
        max_context_terms: int = 10,
        use_embeddings: bool = True,
        embedding_model: Optional[str] = "all-MiniLM-L6-v2"
    ):
        """
        Initialize contextual expander.

        Args:
            max_context_terms: Maximum context terms to add
            use_embeddings: Whether to use embeddings for relevance
            embedding_model: Sentence transformer model
        """
        self.max_context_terms = max_context_terms
        self.use_embeddings = use_embeddings and SENTENCE_TRANSFORMERS_AVAILABLE

        if self.use_embeddings:
            try:
                self.embedding_model = SentenceTransformer(embedding_model)
            except Exception as e:
                logger.warning(f"Failed to initialize embedding model: {e}")
                self.use_embeddings = False

        self.tokenizer = Tokenizer()

        logger.info(f"ContextualExpander initialized: max_context_terms={max_context_terms}")

    def expand_with_context(
        self,
        query: str,
        context: str,
        mode: ExpansionMode = ExpansionMode.EXTEND
    ) -> ExpansionResult:
        """
        Expand query with context-aware terms.

        Args:
            query: Query string
            context: Context text (e.g., conversation history, document preview)
            mode: Expansion mode

        Returns:
            ExpansionResult object
        """
        # Extract important terms from context
        context_terms = self._extract_important_terms(context, query)

        # Score terms by relevance to query
        scored_terms = self._score_terms(query, context_terms)

        # Select top terms
        top_terms = scored_terms[:self.max_context_terms]

        # Build expanded query
        tokens = self.tokenizer.tokenize(query, remove_stopwords=False)
        expanded_tokens = tokens.copy()
        expanded_terms = []
        seen_terms = set(tokens)

        for term, score in top_terms:
            if term not in seen_terms:
                if mode == ExpansionMode.EXTEND or mode == ExpansionMode.BOOST:
                    expanded_tokens.append(term)
                expanded_terms.append(ExpandedTerm(
                    term=term,
                    original_term=query,
                    weight=score,
                    source="context",
                    confidence=score
                ))
                seen_terms.add(term)

        expanded_query = self.tokenizer.detokenize(expanded_tokens)

        return ExpansionResult(
            original_query=query,
            expanded_query=expanded_query,
            expanded_terms=expanded_terms,
            strategy=ExpansionStrategy.CONTEXT,
            mode=mode,
            confidence=min(1.0, len(top_terms) / self.max_context_terms),
            metadata={
                "context_length": len(context),
                "num_context_terms": len(context_terms),
                "selected_terms": len(top_terms)
            }
        )

    def _extract_important_terms(self, context: str, query: str) -> List[str]:
        """Extract important terms from context."""
        # Tokenize context
        tokens = self.tokenizer.tokenize(context, remove_stopwords=True)

        # Remove query terms (they're already known)
        query_tokens = set(self.tokenizer.tokenize(query, remove_stopwords=True))

        # Count term frequencies
        freq = defaultdict(int)
        for token in tokens:
            if token not in query_tokens and len(token) > 2:
                freq[token] += 1

        # Sort by frequency
        sorted_terms = sorted(freq.items(), key=lambda x: x[1], reverse=True)

        return [term for term, _ in sorted_terms]

    def _score_terms(self, query: str, terms: List[str]) -> List[Tuple[str, float]]:
        """Score terms by relevance to query."""
        if not terms:
            return []

        if self.use_embeddings:
            try:
                # Encode query and terms
                query_emb = self.embedding_model.encode([query])[0]
                term_embs = self.embedding_model.encode(terms)

                # Compute similarities
                similarities = util.cos_sim(query_emb, term_embs)[0]

                # Score terms
                scored = []
                for i, term in enumerate(terms):
                    score = float(similarities[i].item())
                    scored.append((term, score))

                # Sort by score
                scored.sort(key=lambda x: x[1], reverse=True)
                return scored

            except Exception as e:
                logger.debug(f"Embedding scoring failed: {e}")

        # Fallback: TF-IDF scoring
        try:
            if SKLEARN_AVAILABLE:
                vectorizer = TfidfVectorizer()
                texts = [query] + terms
                tfidf = vectorizer.fit_transform(texts)

                # Query vector
                query_vec = tfidf[0]
                term_vecs = tfidf[1:]

                # Compute similarities
                similarities = cosine_similarity(query_vec, term_vecs)[0]

                scored = []
                for i, term in enumerate(terms):
                    scored.append((term, float(similarities[i])))

                scored.sort(key=lambda x: x[1], reverse=True)
                return scored

        except Exception as e:
            logger.debug(f"TF-IDF scoring failed: {e}")

        # Simple fallback: use term frequency
        return [(term, 0.5) for term in terms[:20]]


class QueryDecomposer:
    """
    Query decomposition for complex multi-part queries.
    Splits query into sub-queries and expands each.
    """

    def __init__(
        self,
        max_subqueries: int = 3,
        use_nlp: bool = True
    ):
        """
        Initialize query decomposer.

        Args:
            max_subqueries: Maximum number of subqueries
            use_nlp: Whether to use NLP for decomposition
        """
        self.max_subqueries = max_subqueries
        self.use_nlp = use_nlp and SPACY_AVAILABLE

        if self.use_nlp:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("spaCy model not found. Run: python -m spacy download en_core_web_sm")
                self.use_nlp = False

        self.tokenizer = Tokenizer()

        logger.info(f"QueryDecomposer initialized: max_subqueries={max_subqueries}")

    def decompose_query(self, query: str) -> ExpansionResult:
        """
        Decompose query into sub-queries.

        Args:
            query: Query string

        Returns:
            ExpansionResult with sub-queries
        """
        # Extract main components
        subqueries = self._extract_subqueries(query)

        # Build expanded query
        expanded_query = " ".join(subqueries[:self.max_subqueries])

        # Create expanded terms
        expanded_terms = []
        for i, subquery in enumerate(subqueries[:self.max_subqueries]):
            expanded_terms.append(ExpandedTerm(
                term=subquery,
                original_term=query,
                weight=1.0 - (i * 0.1),
                source=f"subquery_{i+1}",
                confidence=0.8
            ))

        return ExpansionResult(
            original_query=query,
            expanded_query=expanded_query,
            expanded_terms=expanded_terms,
            strategy=ExpansionStrategy.DECOMPOSITION,
            mode=ExpansionMode.EXTEND,
            confidence=min(1.0, len(subqueries) / 3),
            metadata={
                "num_subqueries": len(subqueries),
                "subqueries": subqueries[:self.max_subqueries]
            }
        )

    def _extract_subqueries(self, query: str) -> List[str]:
        """Extract sub-queries from a complex query."""
        subqueries = []

        # Split by question words
        question_words = ['what', 'when', 'where', 'who', 'why', 'how', 'which']

        # Method 1: Split by conjunctions
        conjunctions = [' and ', ' or ', ' but ', ', ']
        parts = [query]

        for conj in conjunctions:
            new_parts = []
            for part in parts:
                if conj in part:
                    split_parts = part.split(conj)
                    new_parts.extend([p.strip() for p in split_parts])
                else:
                    new_parts.append(part)
            parts = new_parts

        # Clean and filter parts
        for part in parts:
            if len(part.split()) > 1 and part:
                subqueries.append(part)

        # Method 2: Use NLP if available
        if self.use_nlp:
            try:
                doc = self.nlp(query)

                # Extract noun chunks and phrases
                for chunk in doc.noun_chunks:
                    if len(chunk.text.split()) > 1:
                        subquery = f"what is {chunk.text}"
                        if subquery not in subqueries:
                            subqueries.append(subquery)

            except Exception as e:
                logger.debug(f"NLP decomposition failed: {e}")

        # Remove duplicates
        subqueries = list(dict.fromkeys(subqueries))

        # Limit number of subqueries
        return subqueries[:self.max_subqueries * 2]


class FAQExpander:
    """
    Query expansion using frequently asked questions (FAQ) patterns.
    """

    def __init__(self, faq_data: Optional[List[Dict[str, str]]] = None):
        """
        Initialize FAQ expander.

        Args:
            faq_data: List of FAQ entries with 'question' and 'answer' fields
        """
        self.faq_data = faq_data or []
        self.tokenizer = Tokenizer()

        # Common FAQ patterns
        self.faq_patterns = [
            (r'^(what|how|why|when|where|who|which)', 'question'),
            (r'^explain', 'explain'),
            (r'^define', 'define'),
            (r'^tell me about', 'explain'),
        ]

        logger.info(f"FAQExpander initialized with {len(self.faq_data)} FAQs")

    def expand_query(self, query: str) -> ExpansionResult:
        """
        Expand query using FAQ patterns.

        Args:
            query: Query string

        Returns:
            ExpansionResult object
        """
        # Find matching FAQ entries
        matches = self._find_matching_faqs(query)

        expanded_terms = []
        expanded_tokens = self.tokenizer.tokenize(query, remove_stopwords=False)

        for i, (faq_question, faq_answer) in enumerate(matches[:3]):
            # Extract important terms from FAQ
            faq_terms = self._extract_terms_from_faq(faq_question, faq_answer)

            for term in faq_terms[:3]:
                expanded_terms.append(ExpandedTerm(
                    term=term,
                    original_term=query,
                    weight=0.7 - (i * 0.1),
                    source=f"faq_{i+1}",
                    confidence=0.6
                ))
                expanded_tokens.append(term)

        # Build expanded query
        expanded_query = self.tokenizer.detokenize(expanded_tokens)

        return ExpansionResult(
            original_query=query,
            expanded_query=expanded_query,
            expanded_terms=expanded_terms,
            strategy=ExpansionStrategy.FAQ,
            mode=ExpansionMode.EXTEND,
            confidence=min(1.0, len(matches) / 2),
            metadata={
                "faq_matches": len(matches),
                "matched_questions": [q for q, _ in matches[:3]]
            }
        )

    def _find_matching_faqs(self, query: str) -> List[Tuple[str, str]]:
        """Find FAQ entries matching the query."""
        if not self.faq_data:
            return []

        query_lower = query.lower()
        query_tokens = set(self.tokenizer.tokenize(query, remove_stopwords=True))

        matches = []
        for faq in self.faq_data:
            faq_question = faq.get('question', '')
            faq_answer = faq.get('answer', '')

            # Calculate similarity
            faq_tokens = set(self.tokenizer.tokenize(faq_question, remove_stopwords=True))
            overlap = len(query_tokens & faq_tokens)

            if overlap > 0:
                score = overlap / max(len(query_tokens), len(faq_tokens))
                if score > 0.2:
                    matches.append((faq_question, faq_answer))

        # Sort by similarity
        matches.sort(key=lambda x: len(set(self.tokenizer.tokenize(x[0], True)) & query_tokens), reverse=True)

        return matches

    def _extract_terms_from_faq(self, question: str, answer: str) -> List[str]:
        """Extract important terms from FAQ entry."""
        combined = f"{question} {answer}"
        tokens = self.tokenizer.tokenize(combined, remove_stopwords=True)

        # Count frequencies
        freq = defaultdict(int)
        for token in tokens:
            if len(token) > 2:
                freq[token] += 1

        # Get top terms
        sorted_terms = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        return [term for term, _ in sorted_terms[:10]]


class EmbeddingExpander:
    """
    Query expansion using embedding-based similar terms.
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        max_expansions: int = 5,
        similarity_threshold: float = 0.7,
        vocabulary: Optional[List[str]] = None
    ):
        """
        Initialize embedding expander.

        Args:
            embedding_model: Sentence transformer model
            max_expansions: Maximum expansions per term
            similarity_threshold: Minimum similarity threshold
            vocabulary: Optional vocabulary of terms
        """
        self.max_expansions = max_expansions
        self.similarity_threshold = similarity_threshold
        self.vocabulary = vocabulary or []

        self.tokenizer = Tokenizer()

        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers not installed")

        try:
            self.embedding_model = SentenceTransformer(embedding_model)
            self.term_embeddings = None
            self.terms = []

            if self.vocabulary:
                self._build_term_index()

        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            raise

        logger.info(f"EmbeddingExpander initialized with {len(self.vocabulary)} terms")

    def _build_term_index(self):
        """Build index of term embeddings."""
        if not self.vocabulary:
            return

        try:
            self.term_embeddings = self.embedding_model.encode(self.vocabulary)
            self.terms = self.vocabulary
        except Exception as e:
            logger.error(f"Failed to build term index: {e}")
            self.term_embeddings = None
            self.terms = []

    def expand_query(self, query: str, mode: ExpansionMode = ExpansionMode.EXTEND) -> ExpansionResult:
        """
        Expand query using embedding similarities.

        Args:
            query: Query string
            mode: Expansion mode

        Returns:
            ExpansionResult object
        """
        if not self.term_embeddings or not self.terms:
            return ExpansionResult(
                original_query=query,
                expanded_query=query,
                expanded_terms=[],
                strategy=ExpansionStrategy.EMBEDDING,
                mode=mode,
                confidence=0.0,
                metadata={"error": "No vocabulary available"}
            )

        tokens = self.tokenizer.tokenize(query, remove_stopwords=False)

        expanded_terms = []
        expanded_tokens = tokens.copy()
        seen_terms = set(tokens)

        for token in tokens:
            if token not in self.terms:
                continue

            # Find similar terms
            try:
                token_idx = self.terms.index(token)
                token_emb = self.term_embeddings[token_idx]

                # Compute similarities
                similarities = util.cos_sim(token_emb, self.term_embeddings)[0]

                # Get top similar terms
                top_indices = similarities.argsort(descending=True)

                for idx in top_indices[1:self.max_expansions + 1]:
                    term = self.terms[idx]
                    score = float(similarities[idx].item())

                    if score >= self.similarity_threshold and term not in seen_terms:
                        expanded_terms.append(ExpandedTerm(
                            term=term,
                            original_term=token,
                            weight=score,
                            source="embedding",
                            confidence=score
                        ))

                        if mode == ExpansionMode.EXTEND or mode == ExpansionMode.BOOST:
                            expanded_tokens.append(term)
                            seen_terms.add(term)

            except Exception as e:
                logger.debug(f"Embedding expansion error for token '{token}': {e}")

        # Build expanded query
        expanded_query = self.tokenizer.detokenize(expanded_tokens)

        return ExpansionResult(
            original_query=query,
            expanded_query=expanded_query,
            expanded_terms=expanded_terms,
            strategy=ExpansionStrategy.EMBEDDING,
            mode=mode,
            confidence=min(1.0, len(expanded_terms) / (len(tokens) + 1)),
            metadata={
                "original_tokens": tokens,
                "expanded_tokens": expanded_tokens,
                "num_expansions": len(expanded_terms)
            }
        )


class QueryExpander:
    """
    Main query expansion pipeline combining multiple strategies.
    """

    def __init__(
        self,
        strategies: List[ExpansionStrategy] = None,
        mode: ExpansionMode = ExpansionMode.EXTEND,
        max_expansions: int = 10,
        use_synonyms: bool = True,
        use_embeddings: bool = True,
        use_context: bool = True,
        use_decomposition: bool = True,
        use_faq: bool = True,
        embedding_model: str = "all-MiniLM-L6-v2",
        faq_data: Optional[List[Dict[str, str]]] = None,
        vocabulary: Optional[List[str]] = None
    ):
        """
        Initialize query expander.

        Args:
            strategies: List of strategies to use
            mode: Expansion mode
            max_expansions: Maximum number of expansions
            use_synonyms: Whether to use synonym expansion
            use_embeddings: Whether to use embedding expansion
            use_context: Whether to use context expansion
            use_decomposition: Whether to use query decomposition
            use_faq: Whether to use FAQ expansion
            embedding_model: Embedding model for expansions
            faq_data: FAQ data for FAQ expansion
            vocabulary: Vocabulary for embedding expansion
        """
        self.mode = mode
        self.max_expansions = max_expansions

        # Initialize strategies
        self.strategies = {}

        if use_synonyms:
            try:
                self.strategies[ExpansionStrategy.SYNONYM] = SynonymExpander(
                    use_wordnet=True,
                    use_embeddings=use_embeddings,
                    embedding_model=embedding_model
                )
            except Exception as e:
                logger.warning(f"Failed to initialize synonym expander: {e}")

        if use_embeddings and vocabulary:
            try:
                self.strategies[ExpansionStrategy.EMBEDDING] = EmbeddingExpander(
                    embedding_model=embedding_model,
                    vocabulary=vocabulary
                )
            except Exception as e:
                logger.warning(f"Failed to initialize embedding expander: {e}")

        if use_context:
            self.strategies[ExpansionStrategy.CONTEXT] = ContextualExpander(
                use_embeddings=use_embeddings,
                embedding_model=embedding_model
            )

        if use_decomposition:
            self.strategies[ExpansionStrategy.DECOMPOSITION] = QueryDecomposer()

        if use_faq and faq_data:
            self.strategies[ExpansionStrategy.FAQ] = FAQExpander(faq_data)

        # Filter strategies if specified
        if strategies:
            self.strategies = {s: self.strategies[s] for s in strategies if s in self.strategies}

        self.tokenizer = Tokenizer()

        logger.info(f"QueryExpander initialized with {len(self.strategies)} strategies")

    def expand(
        self,
        query: str,
        context: Optional[str] = None,
        mode: Optional[ExpansionMode] = None,
        strategy: Optional[ExpansionStrategy] = None
    ) -> ExpansionResult:
        """
        Expand query using all available strategies.

        Args:
            query: Query string
            context: Optional context for contextual expansion
            mode: Expansion mode (overrides default)
            strategy: Specific strategy to use

        Returns:
            ExpansionResult object
        """
        if strategy:
            # Use specific strategy
            if strategy not in self.strategies:
                return ExpansionResult(
                    original_query=query,
                    expanded_query=query,
                    expanded_terms=[],
                    strategy=strategy,
                    mode=mode or self.mode,
                    confidence=0.0,
                    metadata={"error": f"Strategy {strategy.value} not available"}
                )

            expander = self.strategies[strategy]
            if strategy == ExpansionStrategy.CONTEXT and context:
                return expander.expand_with_context(query, context, mode or self.mode)
            else:
                return expander.expand_query(query, mode or self.mode)

        # Use all strategies
        all_expanded_terms = []
        all_results = []

        for strat, expander in self.strategies.items():
            try:
                if strat == ExpansionStrategy.CONTEXT and context:
                    result = expander.expand_with_context(query, context, mode or self.mode)
                else:
                    result = expander.expand_query(query, mode or self.mode)

                all_results.append(result)
                all_expanded_terms.extend(result.expanded_terms)

            except Exception as e:
                logger.warning(f"Strategy {strat.value} failed: {e}")

        if not all_expanded_terms:
            return ExpansionResult(
                original_query=query,
                expanded_query=query,
                expanded_terms=[],
                strategy=ExpansionStrategy.HYBRID,
                mode=mode or self.mode,
                confidence=0.0,
                metadata={"strategies_used": list(self.strategies.keys())}
            )

        # Deduplicate and weight terms
        unique_terms = {}
        for term in all_expanded_terms:
            key = term.term.lower()
            if key not in unique_terms or term.weight > unique_terms[key].weight:
                unique_terms[key] = term

        # Sort by weight
        sorted_terms = sorted(unique_terms.values(), key=lambda x: x.weight, reverse=True)
        selected_terms = sorted_terms[:self.max_expansions]

        # Build expanded query
        tokens = self.tokenizer.tokenize(query, remove_stopwords=False)
        expanded_tokens = tokens.copy()
        seen_terms = set(tokens)

        for term in selected_terms:
            if term.term.lower() not in seen_terms:
                if mode == ExpansionMode.EXTEND or mode == ExpansionMode.BOOST:
                    expanded_tokens.append(term.term)
                seen_terms.add(term.term.lower())

        expanded_query = self.tokenizer.detokenize(expanded_tokens)

        # Calculate overall confidence
        confidence = min(1.0, len(selected_terms) / max(1, len(tokens)))

        return ExpansionResult(
            original_query=query,
            expanded_query=expanded_query,
            expanded_terms=selected_terms,
            strategy=ExpansionStrategy.HYBRID,
            mode=mode or self.mode,
            confidence=confidence,
            metadata={
                "strategies_used": [r.strategy.value for r in all_results],
                "total_terms_before": len(all_expanded_terms),
                "total_terms_after": len(selected_terms)
            }
        )

    def expand_batch(
        self,
        queries: List[str],
        contexts: Optional[List[str]] = None
    ) -> List[ExpansionResult]:
        """
        Expand multiple queries.

        Args:
            queries: List of queries
            contexts: Optional list of contexts

        Returns:
            List of ExpansionResult objects
        """
        results = []
        for i, query in enumerate(queries):
            context = contexts[i] if contexts and i < len(contexts) else None
            results.append(self.expand(query, context))
        return results


# Convenience functions
def expand_query(
    query: str,
    context: Optional[str] = None,
    use_synonyms: bool = True,
    use_embeddings: bool = True,
    use_context: bool = True,
    use_decomposition: bool = True,
    max_expansions: int = 10
) -> ExpansionResult:
    """
    Quick query expansion with default settings.

    Args:
        query: Query string
        context: Optional context
        use_synonyms: Whether to use synonyms
        use_embeddings: Whether to use embeddings
        use_context: Whether to use context
        use_decomposition: Whether to decompose query
        max_expansions: Maximum expansions

    Returns:
        ExpansionResult object
    """
    expander = QueryExpander(
        use_synonyms=use_synonyms,
        use_embeddings=use_embeddings,
        use_context=use_context,
        use_decomposition=use_decomposition,
        max_expansions=max_expansions
    )

    return expander.expand(query, context)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Test queries
    queries = [
        "What is machine learning?",
        "How do neural networks work?",
        "Tell me about deep learning and its applications.",
        "What are the benefits of using RAG for document QA?"
    ]

    print("Testing Query Expansion...")
    print("=" * 60)

    # Create expander
    expander = QueryExpander(
        use_synonyms=True,
        use_embeddings=False,  # Disable for demo
        use_context=True,
        use_decomposition=True,
        use_faq=False,
        max_expansions=8
    )

    for query in queries:
        print(f"\nOriginal: {query}")

        # Expand query
        result = expander.expand(query, context="This is about artificial intelligence and machine learning techniques.")

        print(f"Expanded: {result.expanded_query}")
        print(f"Strategy: {result.strategy.value}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Expanded terms: {len(result.expanded_terms)}")

        if result.expanded_terms:
            print("Top expanded terms:")
            for term in result.expanded_terms[:5]:
                print(f"  - {term.term} (weight: {term.weight:.2f}, source: {term.source})")

        print("-" * 40)

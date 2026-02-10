"""
Faithfulness scoring module for evaluating factual consistency of generated responses.
Provides multiple approaches for detecting hallucinations and measuring faithfulness:
- NLI-based entailment scoring
- Token-level alignment
- Fact extraction and verification
- Contradiction detection
- Confidence scoring
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple, Union, Set
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import numpy as np
from functools import lru_cache
import math

# Try importing NLP libraries
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logging.warning("spacy not installed. Install with: pip install spacy && python -m spacy download en_core_web_sm")

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("transformers not installed. Install with: pip install transformers")

try:
    from sentence_transformers import SentenceTransformer, util
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers not installed. Install with: pip install sentence-transformers")

logger = logging.getLogger(__name__)


@dataclass
class FaithfulnessResult:
    """Result of faithfulness scoring."""
    score: float  # 0-1 score, 1 = fully faithful
    is_faithful: bool
    confidence: float
    explanations: List[str] = field(default_factory=list)
    contradictions: List[Dict[str, Any]] = field(default_factory=list)
    supported_claims: List[Dict[str, Any]] = field(default_factory=list)
    unsupported_claims: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "score": self.score,
            "is_faithful": self.is_faithful,
            "confidence": self.confidence,
            "explanations": self.explanations,
            "contradictions": self.contradictions,
            "supported_claims": self.supported_claims[:5],  # Limit for display
            "unsupported_claims": self.unsupported_claims[:5],  # Limit for display
            "metadata": self.metadata
        }


class ClaimExtractor:
    """Extract atomic claims from text for fact verification."""

    def __init__(self, use_spacy: bool = True):
        """
        Initialize claim extractor.

        Args:
            use_spacy: Use spaCy for better extraction
        """
        self.use_spacy = use_spacy and SPACY_AVAILABLE

        if self.use_spacy:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("spaCy model not found. Run: python -m spacy download en_core_web_sm")
                self.use_spacy = False

        # Patterns for extracting factual claims
        self.claim_patterns = [
            r'(?P<subject>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:is|are|was|were|has|have|will|would|could|should|may|might)\s+(?P<claim>.+?)[.!?]',
            r'(?P<subject>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:does|did)\s+(?:not\s+)?(?P<claim>.+?)[.!?]',
            r'The\s+(?P<subject>.+?)\s+(?:is|are|was|were|has|have)\s+(?P<claim>.+?)[.!?]',
            r'(?P<subject>.+?)\s+(?:was|were)\s+(?:founded|established|created|developed)\s+(?P<claim>.+?)[.!?]',
        ]

        # Temporal expressions
        self.temporal_patterns = [
            r'\b(?:in|on|at)\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}',
            r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',
            r'\b\d{4}\b',
            r'\b(?:yesterday|today|tomorrow|last\s+\w+|next\s+\w+)\b',
        ]

        # Numerical expressions
        self.numerical_patterns = [
            r'\b\d+\.?\d*\s*%',
            r'\$\d+(?:,\d+)*(?:\.\d+)?',
            r'\b\d+(?:,\d+)*\b',
        ]

    def extract_claims(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract atomic factual claims from text.

        Args:
            text: Text to extract claims from

        Returns:
            List of extracted claims with metadata
        """
        claims = []

        # Split into sentences
        sentences = self._split_sentences(text)

        for sent_idx, sentence in enumerate(sentences):
            # Extract claims using patterns
            for pattern in self.claim_patterns:
                matches = re.finditer(pattern, sentence, re.IGNORECASE)
                for match in matches:
                    claim = {
                        "text": match.group(0),
                        "sentence": sentence,
                        "sentence_index": sent_idx,
                        "subject": match.group('subject') if 'subject' in match.groupdict() else None,
                        "claim": match.group('claim') if 'claim' in match.groupdict() else None,
                        "type": "factual",
                        "temporal": self._has_temporal(sentence),
                        "numerical": self._has_numerical(sentence)
                    }
                    claims.append(claim)
                    break  # Use first pattern that matches

        # If no claims found, treat each sentence as a claim
        if not claims:
            for sent_idx, sentence in enumerate(sentences):
                if len(sentence.split()) > 5:  # Only meaningful sentences
                    claims.append({
                        "text": sentence,
                        "sentence": sentence,
                        "sentence_index": sent_idx,
                        "subject": None,
                        "claim": sentence,
                        "type": "statement",
                        "temporal": self._has_temporal(sentence),
                        "numerical": self._has_numerical(sentence)
                    })

        return claims

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        if self.use_spacy:
            doc = self.nlp(text)
            return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        else:
            # Simple sentence splitting
            sentences = re.split(r'[.!?]+\s+', text)
            return [s.strip() for s in sentences if s.strip()]

    def _has_temporal(self, text: str) -> bool:
        """Check if text contains temporal expressions."""
        for pattern in self.temporal_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    def _has_numerical(self, text: str) -> bool:
        """Check if text contains numerical expressions."""
        for pattern in self.numerical_patterns:
            if re.search(pattern, text):
                return True
        return False


class NLIEntailmentScorer:
    """
    NLI-based entailment scoring for faithfulness evaluation.
    Uses pre-trained NLI models to check if generated text is entailed by source.
    """

    SUPPORTED_MODELS = {
        "bart-large-mnli": "facebook/bart-large-mnli",
        "roberta-large-mnli": "roberta-large-mnli",
        "albert-xxlarge-v2-mnli": "albert-xxlarge-v2-mnli",
        "electra-large-discriminator": "google/electra-large-discriminator",
        "deberta-v3-base": "microsoft/deberta-v3-base",
    }

    def __init__(
        self,
        model_name: str = "bart-large-mnli",
        device: str = "cpu",
        batch_size: int = 32,
        threshold: float = 0.5
    ):
        """
        Initialize NLI entailment scorer.

        Args:
            model_name: Name of the NLI model
            device: Device to run model on ('cpu' or 'cuda')
            batch_size: Batch size for inference
            threshold: Entailment threshold
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers not installed. Install with: pip install transformers")

        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.threshold = threshold

        # Get model path
        model_path = self.SUPPORTED_MODELS.get(model_name, model_name)

        # Initialize pipeline
        self.pipeline = pipeline(
            "text-classification",
            model=model_path,
            device=0 if device == "cuda" else -1,
            batch_size=batch_size
        )

        logger.info(f"Initialized NLIEntailmentScorer with model={model_name}, device={device}")

    def score(
        self,
        generated: str,
        source: str,
        granularity: str = "sentence"  # "sentence", "claim", "document"
    ) -> FaithfulnessResult:
        """
        Score faithfulness using NLI entailment.

        Args:
            generated: Generated text to evaluate
            source: Source text to check against
            granularity: Level of granularity ('sentence', 'claim', 'document')

        Returns:
            FaithfulnessResult object
        """
        if not generated or not source:
            return FaithfulnessResult(
                score=0.0,
                is_faithful=False,
                confidence=0.0,
                explanations=["Empty text provided"]
            )

        # Extract claims or sentences
        if granularity == "claim":
            extractor = ClaimExtractor()
            generated_units = extractor.extract_claims(generated)
            generated_texts = [c["text"] for c in generated_units]
        elif granularity == "sentence":
            extractor = ClaimExtractor()
            generated_texts = extractor._split_sentences(generated)
        else:  # document level
            generated_texts = [generated]

        if not generated_texts:
            return FaithfulnessResult(
                score=0.0,
                is_faithful=False,
                confidence=0.0,
                explanations=["No claims or sentences extracted"]
            )

        # Prepare pairs for NLI
        pairs = [(source, text) for text in generated_texts]

        try:
            # Get NLI predictions
            results = self.pipeline(pairs)

            # Process results
            entailment_scores = []
            supported_claims = []
            unsupported_claims = []
            contradictions = []
            explanations = []

            for i, result in enumerate(results):
                if isinstance(result, list):
                    result = result[0]

                label = result['label']
                score = result['score']

                # Map to faithfulness score
                if label == 'ENTAILMENT':
                    faith_score = score
                    supported_claims.append({
                        "text": generated_texts[i],
                        "score": score,
                        "label": label
                    })
                    explanations.append(f"Claim '{generated_texts[i][:50]}...' is entailed by source")
                elif label == 'NEUTRAL':
                    faith_score = 0.5 + (score - 0.5) * 0.5
                    unsupported_claims.append({
                        "text": generated_texts[i],
                        "score": score,
                        "label": label
                    })
                    explanations.append(f"Claim '{generated_texts[i][:50]}...' is neutral (not verifiable)")
                else:  # CONTRADICTION
                    faith_score = 1 - score
                    contradictions.append({
                        "text": generated_texts[i],
                        "score": score,
                        "label": label
                    })
                    explanations.append(f"Claim '{generated_texts[i][:50]}...' contradicts source")

                entailment_scores.append(faith_score)

            # Calculate overall score
            overall_score = np.mean(entailment_scores) if entailment_scores else 0.0
            is_faithful = overall_score >= self.threshold

            # Calculate confidence
            confidence = 1.0 - np.std(entailment_scores) if len(entailment_scores) > 1 else overall_score
            confidence = max(0.0, min(1.0, confidence))

            return FaithfulnessResult(
                score=overall_score,
                is_faithful=is_faithful,
                confidence=confidence,
                explanations=explanations[:10],  # Limit explanations
                contradictions=contradictions,
                supported_claims=supported_claims,
                unsupported_claims=unsupported_claims,
                metadata={
                    "model": self.model_name,
                    "granularity": granularity,
                    "num_claims": len(generated_texts)
                }
            )

        except Exception as e:
            logger.warning(f"NLI scoring failed: {e}")
            return FaithfulnessResult(
                score=0.0,
                is_faithful=False,
                confidence=0.0,
                explanations=[f"Scoring failed: {str(e)}"]
            )


class SemanticSimilarityScorer:
    """
    Faithfulness scoring using semantic similarity.
    Compares generated text to source using sentence embeddings.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        threshold: float = 0.7,
        aggregation: str = "mean"  # "mean", "max", "min"
    ):
        """
        Initialize semantic similarity scorer.

        Args:
            model_name: Sentence transformer model name
            threshold: Similarity threshold for faithfulness
            aggregation: Method to aggregate scores
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers not installed. Install with: pip install sentence-transformers")

        self.model_name = model_name
        self.threshold = threshold
        self.aggregation = aggregation

        # Initialize model
        self.model = SentenceTransformer(model_name)

        logger.info(f"Initialized SemanticSimilarityScorer with model={model_name}")

    def score(
        self,
        generated: str,
        source: str,
        granularity: str = "sentence"
    ) -> FaithfulnessResult:
        """
        Score faithfulness using semantic similarity.

        Args:
            generated: Generated text to evaluate
            source: Source text to check against
            granularity: Level of granularity ('sentence', 'paragraph')

        Returns:
            FaithfulnessResult object
        """
        if not generated or not source:
            return FaithfulnessResult(
                score=0.0,
                is_faithful=False,
                confidence=0.0,
                explanations=["Empty text provided"]
            )

        # Split into units
        extractor = ClaimExtractor()

        if granularity == "sentence":
            generated_units = extractor._split_sentences(generated)
            source_units = extractor._split_sentences(source)
        else:  # paragraph
            generated_units = [generated]
            source_units = [source]

        if not generated_units:
            return FaithfulnessResult(
                score=0.0,
                is_faithful=False,
                confidence=0.0,
                explanations=["No sentences extracted"]
            )

        try:
            # Generate embeddings
            gen_embeddings = self.model.encode(generated_units, convert_to_tensor=True)
            src_embeddings = self.model.encode(source_units, convert_to_tensor=True)

            # Compute similarity scores
            similarity_scores = []
            explanations = []

            for i, gen_emb in enumerate(gen_embeddings):
                # Find best matching source sentence
                scores = util.cos_sim(gen_emb, src_embeddings)[0]
                best_score = float(scores.max().item())
                best_idx = int(scores.argmax().item())

                similarity_scores.append(best_score)
                explanations.append(
                    f"Generated unit '{generated_units[i][:50]}...' "
                    f"matches source with similarity {best_score:.3f}"
                )

            # Aggregate scores
            if self.aggregation == "mean":
                overall_score = np.mean(similarity_scores)
            elif self.aggregation == "max":
                overall_score = np.max(similarity_scores)
            elif self.aggregation == "min":
                overall_score = np.min(similarity_scores)
            else:
                overall_score = np.mean(similarity_scores)

            # Determine faithfulness
            is_faithful = overall_score >= self.threshold

            # Calculate confidence (based on score spread)
            confidence = 1.0 - (np.std(similarity_scores) if len(similarity_scores) > 1 else 0.0)
            confidence = max(0.0, min(1.0, confidence))

            # Identify issues
            contradictions = []
            unsupported_claims = []
            supported_claims = []

            for i, score in enumerate(similarity_scores):
                if score < 0.4:
                    contradictions.append({
                        "text": generated_units[i],
                        "similarity": score
                    })
                elif score < self.threshold:
                    unsupported_claims.append({
                        "text": generated_units[i],
                        "similarity": score
                    })
                else:
                    supported_claims.append({
                        "text": generated_units[i],
                        "similarity": score
                    })

            return FaithfulnessResult(
                score=overall_score,
                is_faithful=is_faithful,
                confidence=confidence,
                explanations=explanations[:10],
                contradictions=contradictions,
                supported_claims=supported_claims,
                unsupported_claims=unsupported_claims,
                metadata={
                    "model": self.model_name,
                    "granularity": granularity,
                    "aggregation": self.aggregation,
                    "num_units": len(generated_units)
                }
            )

        except Exception as e:
            logger.warning(f"Semantic similarity scoring failed: {e}")
            return FaithfulnessResult(
                score=0.0,
                is_faithful=False,
                confidence=0.0,
                explanations=[f"Scoring failed: {str(e)}"]
            )


class TokenOverlapScorer:
    """
    Token overlap-based faithfulness scoring.
    Measures lexical overlap between generated and source text.
    """

    def __init__(
        self,
        threshold: float = 0.3,
        ignore_stopwords: bool = True
    ):
        """
        Initialize token overlap scorer.

        Args:
            threshold: Overlap threshold for faithfulness
            ignore_stopwords: Ignore stopwords in overlap calculation
        """
        self.threshold = threshold
        self.ignore_stopwords = ignore_stopwords

        # Common stopwords
        self.stopwords = {
            'a', 'an', 'the', 'of', 'to', 'for', 'with', 'on', 'at', 'from',
            'by', 'in', 'as', 'is', 'was', 'were', 'are', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'must'
        }

    def _tokenize(self, text: str) -> Set[str]:
        """Tokenize text into set of tokens."""
        # Lowercase and split
        tokens = set(text.lower().split())

        # Remove punctuation
        tokens = {t.strip('.,!?;:()[]{}"\'') for t in tokens}

        # Remove stopwords if enabled
        if self.ignore_stopwords:
            tokens = {t for t in tokens if t not in self.stopwords}

        # Remove empty tokens
        tokens = {t for t in tokens if t}

        return tokens

    def score(
        self,
        generated: str,
        source: str,
        granularity: str = "document"
    ) -> FaithfulnessResult:
        """
        Score faithfulness using token overlap.

        Args:
            generated: Generated text to evaluate
            source: Source text to check against
            granularity: Level of granularity

        Returns:
            FaithfulnessResult object
        """
        if not generated or not source:
            return FaithfulnessResult(
                score=0.0,
                is_faithful=False,
                confidence=0.0,
                explanations=["Empty text provided"]
            )

        # Tokenize
        gen_tokens = self._tokenize(generated)
        src_tokens = self._tokenize(source)

        if not gen_tokens or not src_tokens:
            return FaithfulnessResult(
                score=0.0,
                is_faithful=False,
                confidence=0.0,
                explanations=["No tokens extracted"]
            )

        # Calculate overlap
        overlap = gen_tokens & src_tokens

        # Precision: overlap / gen_tokens
        precision = len(overlap) / len(gen_tokens) if gen_tokens else 0

        # Recall: overlap / src_tokens
        recall = len(overlap) / len(src_tokens) if src_tokens else 0

        # F1 score
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0

        # Additional metrics
        coverage = len(overlap) / max(len(gen_tokens), len(src_tokens)) if max(len(gen_tokens), len(src_tokens)) > 0 else 0

        # Determine faithfulness
        overall_score = f1
        is_faithful = overall_score >= self.threshold

        # Explanations
        explanations = [
            f"Token overlap: {len(overlap)} shared tokens",
            f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}",
            f"Coverage: {coverage:.3f}"
        ]

        # Identify unsupported tokens
        unsupported = gen_tokens - src_tokens
        unsupported_list = list(unsupported)[:20]  # Limit

        return FaithfulnessResult(
            score=overall_score,
            is_faithful=is_faithful,
            confidence=f1,
            explanations=explanations,
            metadata={
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "coverage": coverage,
                "total_tokens_gen": len(gen_tokens),
                "total_tokens_src": len(src_tokens),
                "overlap_tokens": len(overlap),
                "unsupported_tokens_sample": unsupported_list
            }
        )


class FaithfulnessScorer:
    """
    Comprehensive faithfulness scorer combining multiple approaches.
    """

    def __init__(
        self,
        methods: List[str] = ["nli", "semantic", "token"],
        weights: Dict[str, float] = None,
        threshold: float = 0.5,
        nli_model: str = "bart-large-mnli",
        semantic_model: str = "all-MiniLM-L6-v2",
        device: str = "cpu"
    ):
        """
        Initialize comprehensive faithfulness scorer.

        Args:
            methods: Methods to use ('nli', 'semantic', 'token')
            weights: Weights for each method
            threshold: Overall faithfulness threshold
            nli_model: NLI model name
            semantic_model: Semantic similarity model name
            device: Device for models
        """
        self.methods = methods
        self.threshold = threshold

        # Set weights
        self.weights = weights or {
            "nli": 0.5,
            "semantic": 0.3,
            "token": 0.2
        }

        # Normalize weights
        total = sum(self.weights.values())
        self.weights = {k: v/total for k, v in self.weights.items()}

        # Initialize scorers
        self.scorers = {}

        if "nli" in methods:
            try:
                self.scorers["nli"] = NLIEntailmentScorer(
                    model_name=nli_model,
                    device=device
                )
            except Exception as e:
                logger.warning(f"Failed to initialize NLI scorer: {e}")

        if "semantic" in methods:
            try:
                self.scorers["semantic"] = SemanticSimilarityScorer(
                    model_name=semantic_model
                )
            except Exception as e:
                logger.warning(f"Failed to initialize semantic scorer: {e}")

        if "token" in methods:
            self.scorers["token"] = TokenOverlapScorer()

        logger.info(f"Initialized FaithfulnessScorer with methods={methods}")

    def score(
        self,
        generated: str,
        source: str,
        granularity: str = "sentence"
    ) -> FaithfulnessResult:
        """
        Score faithfulness using multiple methods.

        Args:
            generated: Generated text to evaluate
            source: Source text to check against
            granularity: Level of granularity

        Returns:
            FaithfulnessResult object
        """
        if not generated or not source:
            return FaithfulnessResult(
                score=0.0,
                is_faithful=False,
                confidence=0.0,
                explanations=["Empty text provided"]
            )

        results = []
        all_explanations = []
        all_contradictions = []
        all_supported = []
        all_unsupported = []
        combined_score = 0.0
        total_weight = 0.0

        for method, scorer in self.scorers.items():
            try:
                result = scorer.score(generated, source, granularity)

                weight = self.weights.get(method, 0.2)
                combined_score += result.score * weight
                total_weight += weight

                results.append(result)
                all_explanations.extend(result.explanations)
                all_contradictions.extend(result.contradictions)
                all_supported.extend(result.supported_claims)
                all_unsupported.extend(result.unsupported_claims)

            except Exception as e:
                logger.warning(f"Method {method} failed: {e}")

        # Normalize combined score
        if total_weight > 0:
            combined_score = combined_score / total_weight
        else:
            combined_score = 0.0

        # Determine faithfulness
        is_faithful = combined_score >= self.threshold

        # Calculate confidence
        if results:
            confidence = np.mean([r.confidence for r in results])
        else:
            confidence = 0.0

        # Combine explanations
        unique_explanations = list(dict.fromkeys(all_explanations))

        return FaithfulnessResult(
            score=combined_score,
            is_faithful=is_faithful,
            confidence=confidence,
            explanations=unique_explanations[:15],
            contradictions=all_contradictions[:10],
            supported_claims=all_supported[:10],
            unsupported_claims=all_unsupported[:10],
            metadata={
                "methods": self.methods,
                "weights": self.weights,
                "granularity": granularity,
                "num_methods": len(self.scorers)
            }
        )


# ============================================================
# Convenience Functions
# ============================================================

def evaluate_faithfulness(
    generated: str,
    source: str,
    method: str = "ensemble",
    threshold: float = 0.5,
    granularity: str = "sentence"
) -> FaithfulnessResult:
    """
    Quick function to evaluate faithfulness of generated text.

    Args:
        generated: Generated text to evaluate
        source: Source text to check against
        method: Method to use ('nli', 'semantic', 'token', 'ensemble')
        threshold: Faithfulness threshold
        granularity: Level of granularity

    Returns:
        FaithfulnessResult object
    """
    if method == "ensemble":
        scorer = FaithfulnessScorer(
            methods=["nli", "semantic", "token"],
            threshold=threshold
        )
    elif method == "nli":
        scorer = NLIEntailmentScorer(threshold=threshold)
    elif method == "semantic":
        scorer = SemanticSimilarityScorer(threshold=threshold)
    elif method == "token":
        scorer = TokenOverlapScorer(threshold=threshold)
    else:
        raise ValueError(f"Unsupported method: {method}")

    return scorer.score(generated, source, granularity)


def batch_evaluate_faithfulness(
    generated_list: List[str],
    source_list: List[str],
    method: str = "ensemble",
    threshold: float = 0.5,
    granularity: str = "sentence"
) -> List[FaithfulnessResult]:
    """
    Batch evaluate faithfulness for multiple examples.

    Args:
        generated_list: List of generated texts
        source_list: List of source texts
        method: Method to use
        threshold: Faithfulness threshold
        granularity: Level of granularity

    Returns:
        List of FaithfulnessResult objects
    """
    results = []

    for generated, source in zip(generated_list, source_list):
        result = evaluate_faithfulness(
            generated, source, method, threshold, granularity
        )
        results.append(result)

    return results


def aggregate_faithfulness_scores(
    results: List[FaithfulnessResult]
) -> Dict[str, float]:
    """
    Aggregate faithfulness scores from multiple evaluations.

    Args:
        results: List of FaithfulnessResult objects

    Returns:
        Dictionary with aggregated statistics
    """
    if not results:
        return {
            "mean_score": 0.0,
            "std_score": 0.0,
            "min_score": 0.0,
            "max_score": 0.0,
            "faithful_ratio": 0.0,
            "total": 0
        }

    scores = [r.score for r in results]
    is_faithful = [r.is_faithful for r in results]

    return {
        "mean_score": np.mean(scores),
        "std_score": np.std(scores),
        "min_score": np.min(scores),
        "max_score": np.max(scores),
        "faithful_ratio": np.mean(is_faithful),
        "total": len(results)
    }


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Sample texts
    source = """
    Machine learning is a subset of artificial intelligence. 
    It enables systems to learn and improve from experience without being explicitly programmed.
    Deep learning uses neural networks with multiple layers.
    """

    generated_faithful = """
    Machine learning is a subset of artificial intelligence that enables systems to learn from data.
    Deep learning uses neural networks with multiple layers for pattern recognition.
    """

    generated_hallucinated = """
    Machine learning was invented in 1980 by John Smith.
    It uses quantum computers to process data.
    Deep learning requires at least 100 layers to work properly.
    """

    # Test different methods
    print("Testing Faithfulness Scoring...")
    print("=" * 60)

    # NLI-based scoring
    print("\n1. NLI-Based Scoring:")
    result = evaluate_faithfulness(
        generated_faithful,
        source,
        method="nli"
    )
    print(f"  Faithful text score: {result.score:.3f} - {'Faithful' if result.is_faithful else 'Not Faithful'}")

    result = evaluate_faithfulness(
        generated_hallucinated,
        source,
        method="nli"
    )
    print(f"  Hallucinated text score: {result.score:.3f} - {'Faithful' if result.is_faithful else 'Not Faithful'}")
    if result.contradictions:
        print(f"  Contradictions found: {len(result.contradictions)}")
        for c in result.contradictions[:2]:
            print(f"    - {c['text'][:60]}...")

    # Ensemble scoring
    print("\n2. Ensemble Scoring:")
    result = evaluate_faithfulness(
        generated_faithful,
        source,
        method="ensemble"
    )
    print(f"  Faithful text score: {result.score:.3f} - {'Faithful' if result.is_faithful else 'Not Faithful'}")

    result = evaluate_faithfulness(
        generated_hallucinated,
        source,
        method="ensemble"
    )
    print(f"  Hallucinated text score: {result.score:.3f} - {'Faithful' if result.is_faithful else 'Not Faithful'}")
    print(f"  Explanations: {len(result.explanations)}")
    for exp in result.explanations[:3]:
        print(f"    - {exp[:60]}...")

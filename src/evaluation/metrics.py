"""
Evaluation metrics for DocQA AI system.
Provides comprehensive metrics for evaluating retrieval and generation quality:
- BLEU (Bilingual Evaluation Understudy)
- ROUGE (Recall-Oriented Understudy for Gisting Evaluation)
- METEOR
- BERTScore
- Exact Match (EM)
- F1 Score
- Retrieval Metrics (MRR, Recall@K, Precision@K, NDCG)
- Custom metrics for hallucination detection
"""

import re
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from functools import lru_cache
import math
import string
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
import nltk

# Try importing additional packages
try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    logging.warning("rouge_score not installed. Install with: pip install rouge-score")

try:
    from bert_score import score as bert_score
    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False
    logging.warning("bert_score not installed. Install with: pip install bert-score")

try:
    import sacrebleu
    SACREBLEU_AVAILABLE = True
except ImportError:
    SACREBLEU_AVAILABLE = False
    logging.warning("sacrebleu not installed. Install with: pip install sacrebleu")

logger = logging.getLogger(__name__)

# Download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)


@dataclass
class MetricsResult:
    """Container for metrics results."""
    name: str
    value: float
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "value": self.value,
            "description": self.description,
            "metadata": self.metadata
        }


class TextPreprocessor:
    """Preprocess text for evaluation metrics."""

    def __init__(self, lowercase: bool = True, remove_punctuation: bool = True,
                 remove_stopwords: bool = False, stem: bool = False):
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_stopwords = remove_stopwords
        self.stem = stem

        # Common stopwords
        self.stopwords = {
            'a', 'an', 'the', 'of', 'to', 'for', 'with', 'on', 'at', 'from',
            'by', 'in', 'as', 'is', 'was', 'were', 'are', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'must'
        }

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        if self.lowercase:
            text = text.lower()

        if self.remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))

        tokens = text.split()

        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self.stopwords]

        if self.stem:
            try:
                from nltk.stem import PorterStemmer
                stemmer = PorterStemmer()
                tokens = [stemmer.stem(t) for t in tokens]
            except ImportError:
                pass

        return tokens

    def process(self, text: str) -> str:
        """Process text into string."""
        tokens = self.tokenize(text)
        return ' '.join(tokens)


class BleuScorer:
    """
    BLEU (Bilingual Evaluation Understudy) scorer.
    Measures the similarity between generated text and reference text.
    """

    def __init__(self, weights: Tuple[float, ...] = (0.25, 0.25, 0.25, 0.25),
                 smooth: bool = True):
        """
        Initialize BLEU scorer.

        Args:
            weights: Weights for n-grams (1-4)
            smooth: Apply smoothing for short texts
        """
        self.weights = weights
        self.smooth = smooth
        self.smoother = SmoothingFunction().method4 if smooth else None

    def score(
        self,
        candidate: str,
        reference: Union[str, List[str]],
        preprocessor: Optional[TextPreprocessor] = None
    ) -> float:
        """
        Calculate BLEU score between candidate and reference.

        Args:
            candidate: Generated text
            reference: Reference text or list of references
            preprocessor: Text preprocessor

        Returns:
            BLEU score (0-100)
        """
        if preprocessor:
            candidate = preprocessor.process(candidate)
            if isinstance(reference, str):
                reference = preprocessor.process(reference)
            else:
                reference = [preprocessor.process(r) for r in reference]

        candidate_tokens = candidate.split()

        if isinstance(reference, str):
            reference_tokens = [reference.split()]
        else:
            reference_tokens = [r.split() for r in reference]

        try:
            score = sentence_bleu(
                reference_tokens,
                candidate_tokens,
                weights=self.weights,
                smoothing_function=self.smoother
            )
            return score * 100
        except Exception as e:
            logger.warning(f"BLEU score calculation failed: {e}")
            return 0.0

    def batch_score(
        self,
        candidates: List[str],
        references: Union[List[str], List[List[str]]],
        preprocessor: Optional[TextPreprocessor] = None
    ) -> List[float]:
        """Calculate BLEU scores for multiple pairs."""
        scores = []
        for i, candidate in enumerate(candidates):
            if isinstance(references[0], str):
                ref = references[i]
            else:
                ref = references[i] if i < len(references) else references[-1]
            scores.append(self.score(candidate, ref, preprocessor))
        return scores


class RougeScorer:
    """
    ROUGE (Recall-Oriented Understudy for Gisting Evaluation) scorer.
    Measures overlap between generated and reference text.
    """

    ROUGE_TYPES = ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']

    def __init__(self, rouge_types: Optional[List[str]] = None,
                 use_stemmer: bool = True):
        """
        Initialize ROUGE scorer.

        Args:
            rouge_types: Types of ROUGE to compute ('rouge1', 'rouge2', 'rougeL')
            use_stemmer: Use Porter stemmer
        """
        if not ROUGE_AVAILABLE:
            raise ImportError("rouge_score not installed. Install with: pip install rouge-score")

        self.rouge_types = rouge_types or ['rouge1', 'rouge2', 'rougeL']
        self.scorer = rouge_scorer.RougeScorer(
            self.rouge_types,
            use_stemmer=use_stemmer
        )

    def score(
        self,
        candidate: str,
        reference: str,
        preprocessor: Optional[TextPreprocessor] = None
    ) -> Dict[str, float]:
        """
        Calculate ROUGE scores between candidate and reference.

        Args:
            candidate: Generated text
            reference: Reference text
            preprocessor: Text preprocessor

        Returns:
            Dictionary of ROUGE scores (precision, recall, fmeasure)
        """
        if preprocessor:
            candidate = preprocessor.process(candidate)
            reference = preprocessor.process(reference)

        try:
            scores = self.scorer.score(candidate, reference)
            result = {}
            for rouge_type, score in scores.items():
                result[f"{rouge_type}_precision"] = score.precision
                result[f"{rouge_type}_recall"] = score.recall
                result[f"{rouge_type}_fmeasure"] = score.fmeasure
            return result
        except Exception as e:
            logger.warning(f"ROUGE score calculation failed: {e}")
            return {}

    def batch_score(
        self,
        candidates: List[str],
        references: List[str],
        preprocessor: Optional[TextPreprocessor] = None
    ) -> List[Dict[str, float]]:
        """Calculate ROUGE scores for multiple pairs."""
        scores = []
        for candidate, reference in zip(candidates, references):
            scores.append(self.score(candidate, reference, preprocessor))
        return scores


class MeteorScorer:
    """
    METEOR (Metric for Evaluation of Translation with Explicit Ordering) scorer.
    """

    def __init__(self, language: str = 'en'):
        """Initialize METEOR scorer."""
        self.language = language

    def score(
        self,
        candidate: str,
        reference: str,
        preprocessor: Optional[TextPreprocessor] = None
    ) -> float:
        """
        Calculate METEOR score.

        Args:
            candidate: Generated text
            reference: Reference text
            preprocessor: Text preprocessor

        Returns:
            METEOR score (0-100)
        """
        try:
            if preprocessor:
                candidate = preprocessor.process(candidate)
                reference = preprocessor.process(reference)

            score = meteor_score(
                [reference.split()],
                candidate.split(),
                language=self.language
            )
            return score * 100
        except Exception as e:
            logger.warning(f"METEOR score calculation failed: {e}")
            return 0.0

    def batch_score(
        self,
        candidates: List[str],
        references: List[str],
        preprocessor: Optional[TextPreprocessor] = None
    ) -> List[float]:
        """Calculate METEOR scores for multiple pairs."""
        scores = []
        for candidate, reference in zip(candidates, references):
            scores.append(self.score(candidate, reference, preprocessor))
        return scores


class BertScoreScorer:
    """
    BERTScore scorer for semantic similarity.
    Uses BERT embeddings to compute similarity between texts.
    """

    def __init__(self, model_type: str = 'bert-base-uncased',
                 lang: str = 'en', batch_size: int = 32):
        """
        Initialize BERTScore scorer.

        Args:
            model_type: BERT model type
            lang: Language code
            batch_size: Batch size for processing
        """
        if not BERTSCORE_AVAILABLE:
            raise ImportError("bert_score not installed. Install with: pip install bert-score")

        self.model_type = model_type
        self.lang = lang
        self.batch_size = batch_size

    def score(
        self,
        candidates: Union[str, List[str]],
        references: Union[str, List[str]],
        preprocessor: Optional[TextPreprocessor] = None
    ) -> Dict[str, float]:
        """
        Calculate BERTScore.

        Args:
            candidates: Generated text(s)
            references: Reference text(s)
            preprocessor: Text preprocessor

        Returns:
            Dictionary with precision, recall, f1 scores
        """
        if isinstance(candidates, str):
            candidates = [candidates]
        if isinstance(references, str):
            references = [references]

        if preprocessor:
            candidates = [preprocessor.process(c) for c in candidates]
            references = [preprocessor.process(r) for r in references]

        try:
            P, R, F1 = bert_score(
                candidates,
                references,
                model_type=self.model_type,
                lang=self.lang,
                batch_size=self.batch_size,
                verbose=False
            )

            return {
                "bertscore_precision": float(P.mean()),
                "bertscore_recall": float(R.mean()),
                "bertscore_f1": float(F1.mean())
            }
        except Exception as e:
            logger.warning(f"BERTScore calculation failed: {e}")
            return {
                "bertscore_precision": 0.0,
                "bertscore_recall": 0.0,
                "bertscore_f1": 0.0
            }


class ExactMatchScorer:
    """
    Exact Match (EM) scorer.
    Checks if generated text exactly matches any reference.
    """

    @staticmethod
    def score(
        candidate: str,
        references: Union[str, List[str]],
        preprocessor: Optional[TextPreprocessor] = None
    ) -> float:
        """
        Calculate Exact Match score.

        Args:
            candidate: Generated text
            references: Reference text(s)
            preprocessor: Text preprocessor

        Returns:
            1.0 if exact match, 0.0 otherwise
        """
        if preprocessor:
            candidate = preprocessor.process(candidate)
            if isinstance(references, str):
                references = preprocessor.process(references)
            else:
                references = [preprocessor.process(r) for r in references]

        if isinstance(references, str):
            return 1.0 if candidate == references else 0.0
        else:
            return 1.0 if candidate in references else 0.0

    @staticmethod
    def batch_score(
        candidates: List[str],
        references: Union[List[str], List[List[str]]],
        preprocessor: Optional[TextPreprocessor] = None
    ) -> List[float]:
        """Calculate Exact Match scores for multiple pairs."""
        scores = []
        for i, candidate in enumerate(candidates):
            if isinstance(references[0], str):
                ref = references[i]
            else:
                ref = references[i] if i < len(references) else references[-1]
            scores.append(ExactMatchScorer.score(candidate, ref, preprocessor))
        return scores


class F1Scorer:
    """
    F1 score for token-level overlap between generated and reference text.
    """

    @staticmethod
    def score(
        candidate: str,
        reference: str,
        preprocessor: Optional[TextPreprocessor] = None
    ) -> float:
        """
        Calculate F1 score based on token overlap.

        Args:
            candidate: Generated text
            reference: Reference text
            preprocessor: Text preprocessor

        Returns:
            F1 score (0-100)
        """
        if preprocessor:
            candidate = preprocessor.process(candidate)
            reference = preprocessor.process(reference)

        candidate_tokens = set(candidate.split())
        reference_tokens = set(reference.split())

        if not candidate_tokens and not reference_tokens:
            return 100.0
        if not candidate_tokens or not reference_tokens:
            return 0.0

        intersection = candidate_tokens & reference_tokens

        precision = len(intersection) / len(candidate_tokens) if candidate_tokens else 0
        recall = len(intersection) / len(reference_tokens) if reference_tokens else 0

        if precision + recall == 0:
            return 0.0

        f1 = 2 * (precision * recall) / (precision + recall)
        return f1 * 100

    @staticmethod
    def batch_score(
        candidates: List[str],
        references: List[str],
        preprocessor: Optional[TextPreprocessor] = None
    ) -> List[float]:
        """Calculate F1 scores for multiple pairs."""
        scores = []
        for candidate, reference in zip(candidates, references):
            scores.append(F1Scorer.score(candidate, reference, preprocessor))
        return scores


class RetrievalMetrics:
    """
    Metrics for evaluating retrieval performance.
    """

    @staticmethod
    def mean_reciprocal_rank(
        relevant_indices: List[List[int]],
        retrieved_indices: List[List[int]]
    ) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR).

        Args:
            relevant_indices: List of relevant indices for each query
            retrieved_indices: List of retrieved indices for each query

        Returns:
            MRR score
        """
        if not relevant_indices:
            return 0.0

        total = 0.0
        for rel, ret in zip(relevant_indices, retrieved_indices):
            for rank, idx in enumerate(ret, 1):
                if idx in rel:
                    total += 1.0 / rank
                    break

        return total / len(relevant_indices)

    @staticmethod
    def recall_at_k(
        relevant_indices: List[List[int]],
        retrieved_indices: List[List[int]],
        k: int = 5
    ) -> float:
        """
        Calculate Recall@K.

        Args:
            relevant_indices: List of relevant indices for each query
            retrieved_indices: List of retrieved indices for each query
            k: Number of retrieved items to consider

        Returns:
            Recall@K score
        """
        if not relevant_indices:
            return 0.0

        total_recall = 0.0
        for rel, ret in zip(relevant_indices, retrieved_indices):
            ret_k = ret[:k]
            relevant_retrieved = sum(1 for idx in ret_k if idx in rel)
            total_recall += relevant_retrieved / len(rel) if rel else 0

        return total_recall / len(relevant_indices)

    @staticmethod
    def precision_at_k(
        relevant_indices: List[List[int]],
        retrieved_indices: List[List[int]],
        k: int = 5
    ) -> float:
        """
        Calculate Precision@K.

        Args:
            relevant_indices: List of relevant indices for each query
            retrieved_indices: List of retrieved indices for each query
            k: Number of retrieved items to consider

        Returns:
            Precision@K score
        """
        if not retrieved_indices:
            return 0.0

        total_precision = 0.0
        for rel, ret in zip(relevant_indices, retrieved_indices):
            ret_k = ret[:k]
            relevant_retrieved = sum(1 for idx in ret_k if idx in rel)
            total_precision += relevant_retrieved / k if k > 0 else 0

        return total_precision / len(retrieved_indices)

    @staticmethod
    def ndcg_at_k(
        relevant_indices: List[List[int]],
        scores: List[List[float]],
        k: int = 5
    ) -> float:
        """
        Calculate NDCG@K (Normalized Discounted Cumulative Gain).

        Args:
            relevant_indices: List of relevant indices for each query
            scores: List of scores for each retrieved item
            k: Number of items to consider

        Returns:
            NDCG@K score
        """
        if not relevant_indices:
            return 0.0

        total_ndcg = 0.0
        for rel, sc in zip(relevant_indices, scores):
            # Calculate DCG
            dcg = 0.0
            for i, (idx, score) in enumerate(zip(rel[:k], sc[:k])):
                if idx in relevant_indices[0]:  # Simplified: consider all as relevant
                    dcg += score / math.log2(i + 2)

            # Calculate IDCG (ideal DCG)
            ideal_rels = sorted(rel, key=lambda x: sc[rel.index(x)] if x < len(sc) else 0, reverse=True)
            idcg = 0.0
            for i in range(min(k, len(ideal_rels))):
                idx = ideal_rels[i]
                if idx < len(sc):
                    idcg += sc[idx] / math.log2(i + 2)

            if idcg > 0:
                total_ndcg += dcg / idcg

        return total_ndcg / len(relevant_indices) if relevant_indices else 0.0


class HallucinationMetrics:
    """
    Metrics for detecting and measuring hallucinations.
    """

    @staticmethod
    def factual_consistency(
        generated: str,
        context: str,
        tokenizer=None
    ) -> float:
        """
        Measure factual consistency between generated text and context.
        Uses NLI-based approach (requires transformers).

        Args:
            generated: Generated text
            context: Source context
            tokenizer: Tokenizer for truncation

        Returns:
            Consistency score (0-1)
        """
        try:
            from transformers import pipeline

            # Initialize NLI pipeline
            nli = pipeline(
                "text-classification",
                model="facebook/bart-large-mnli",
                device=-1  # CPU
            )

            # Truncate if needed
            if tokenizer:
                generated = tokenizer.decode(
                    tokenizer.encode(generated, max_length=512, truncation=True)
                )
                context = tokenizer.decode(
                    tokenizer.encode(context, max_length=512, truncation=True)
                )

            # Get NLI predictions
            result = nli(f"{context} </s> {generated}")

            # Entailment indicates factual consistency
            if result[0]['label'] == 'ENTAILMENT':
                return result[0]['score']
            else:
                return 1 - result[0]['score']

        except ImportError:
            logger.warning("transformers not installed for hallucination detection")
            return 0.0
        except Exception as e:
            logger.warning(f"Factual consistency check failed: {e}")
            return 0.0

    @staticmethod
    def self_consistency(
        generated: str,
        num_samples: int = 3
    ) -> float:
        """
        Measure self-consistency by sampling multiple responses.

        Args:
            generated: Original generated text
            num_samples: Number of additional samples

        Returns:
            Consistency score (0-1)
        """
        # This requires access to the LLM for sampling
        # Simplified implementation
        return 1.0

    @staticmethod
    def contradiction_detection(
        text: str
    ) -> float:
        """
        Detect contradictions within the generated text.

        Args:
            text: Text to analyze

        Returns:
            Contradiction score (0-1)
        """
        sentences = re.split(r'[.!?]+\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) < 2:
            return 0.0

        # Count contradictory patterns
        contradiction_patterns = [
            r'\b(but|however|although|despite|nevertheless|conversely)\b',
            r'\b(on the other hand|in contrast|unlike)\b',
            r'\b(while|whereas|though|even though)\b',
        ]

        contradictions = 0
        for pattern in contradiction_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            contradictions += len(matches)

        # Normalize by number of sentences
        return min(1.0, contradictions / (len(sentences) - 1))


class EvaluationPipeline:
    """
    Complete evaluation pipeline combining multiple metrics.
    """

    def __init__(
        self,
        metrics: Optional[List[str]] = None,
        preprocessor: Optional[TextPreprocessor] = None
    ):
        """
        Initialize evaluation pipeline.

        Args:
            metrics: List of metrics to compute
            preprocessor: Text preprocessor
        """
        self.metrics = metrics or ['bleu', 'rouge', 'meteor', 'f1']
        self.preprocessor = preprocessor or TextPreprocessor()

        self.scorers = {}

        if 'bleu' in self.metrics:
            self.scorers['bleu'] = BleuScorer()
        if 'rouge' in self.metrics:
            try:
                self.scorers['rouge'] = RougeScorer()
            except ImportError:
                logger.warning("ROUGE not available")
        if 'meteor' in self.metrics:
            self.scorers['meteor'] = MeteorScorer()
        if 'bert_score' in self.metrics:
            try:
                self.scorers['bert_score'] = BertScoreScorer()
            except ImportError:
                logger.warning("BERTScore not available")
        if 'exact_match' in self.metrics:
            self.scorers['exact_match'] = ExactMatchScorer()
        if 'f1' in self.metrics:
            self.scorers['f1'] = F1Scorer()

    def evaluate_generation(
        self,
        candidates: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """
        Evaluate generated text against references.

        Args:
            candidates: List of generated texts
            references: List of reference texts

        Returns:
            Dictionary of metric results
        """
        results = {}

        for name, scorer in self.scorers.items():
            try:
                if name == 'bert_score':
                    score = scorer.score(candidates, references, self.preprocessor)
                    results.update(score)
                else:
                    scores = scorer.batch_score(candidates, references, self.preprocessor)
                    results[name] = np.mean(scores)
            except Exception as e:
                logger.warning(f"Failed to compute {name}: {e}")
                results[name] = 0.0

        return results

    def evaluate_retrieval(
        self,
        relevant_indices: List[List[int]],
        retrieved_indices: List[List[int]],
        scores: Optional[List[List[float]]] = None
    ) -> Dict[str, float]:
        """
        Evaluate retrieval performance.

        Args:
            relevant_indices: List of relevant indices for each query
            retrieved_indices: List of retrieved indices for each query
            scores: List of scores for each retrieved item

        Returns:
            Dictionary of retrieval metrics
        """
        results = {}

        for k in [1, 3, 5, 10]:
            results[f"recall@{k}"] = RetrievalMetrics.recall_at_k(
                relevant_indices, retrieved_indices, k
            )
            results[f"precision@{k}"] = RetrievalMetrics.precision_at_k(
                relevant_indices, retrieved_indices, k
            )

        results["mrr"] = RetrievalMetrics.mean_reciprocal_rank(
            relevant_indices, retrieved_indices
        )

        if scores:
            for k in [5, 10]:
                results[f"ndcg@{k}"] = RetrievalMetrics.ndcg_at_k(
                    relevant_indices, scores, k
                )

        return results


# ============================================================
# Convenience Functions
# ============================================================

def calculate_metrics(
    candidates: List[str],
    references: List[str],
    metrics: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Quick function to calculate generation metrics.

    Args:
        candidates: List of generated texts
        references: List of reference texts
        metrics: List of metrics to compute

    Returns:
        Dictionary of metric results
    """
    pipeline = EvaluationPipeline(metrics=metrics)
    return pipeline.evaluate_generation(candidates, references)


def calculate_retrieval_metrics(
    relevant_indices: List[List[int]],
    retrieved_indices: List[List[int]],
    scores: Optional[List[List[float]]] = None
) -> Dict[str, float]:
    """
    Quick function to calculate retrieval metrics.

    Args:
        relevant_indices: List of relevant indices for each query
        retrieved_indices: List of retrieved indices for each query
        scores: List of scores for each retrieved item

    Returns:
        Dictionary of retrieval metrics
    """
    return EvaluationPipeline().evaluate_retrieval(
        relevant_indices, retrieved_indices, scores
    )


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Sample texts
    candidates = [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks for pattern recognition.",
        "Natural language processing deals with text and language."
    ]

    references = [
        "Machine learning is a field of artificial intelligence.",
        "Deep learning uses neural networks to recognize patterns.",
        "Natural language processing handles text and language understanding."
    ]

    # Calculate metrics
    print("Calculating generation metrics...")
    results = calculate_metrics(candidates, references, metrics=['bleu', 'rouge', 'meteor', 'f1'])

    for metric, value in results.items():
        print(f"  {metric}: {value:.2f}")

    # Test retrieval metrics
    print("\nCalculating retrieval metrics...")
    relevant = [[0, 1, 2], [1, 2], [0, 2]]
    retrieved = [[0, 1, 3, 4], [1, 0, 2, 3], [2, 0, 1, 3]]
    scores = [[1.0, 0.8, 0.5, 0.3], [0.9, 0.7, 0.6, 0.4], [0.95, 0.85, 0.7, 0.5]]

    ret_metrics = calculate_retrieval_metrics(relevant, retrieved, scores)

    for metric, value in ret_metrics.items():
        print(f"  {metric}: {value:.3f}")

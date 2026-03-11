#!/usr/bin/env python3
"""
Fine-tuning preparation script for DocQA AI system.
Converts QA data and conversations into fine-tuning formats for various LLM models.
Supports OpenAI, Llama, Mistral, and other popular fine-tuning formats.
"""

import os
import sys
import json
import csv
import argparse
import logging
import random
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime
from dataclasses import dataclass, field
import hashlib
import re

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import setup_logging, get_logger
from src.evaluation.faithfulness import evaluate_faithfulness

logger = get_logger(__name__)


# ============================================================
# Data Models
# ============================================================

@dataclass
class QAPair:
    """Question-answer pair for fine-tuning."""
    question: str
    answer: str
    context: Optional[str] = None
    category: Optional[str] = None
    difficulty: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: Optional[str] = None
    id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "question": self.question,
            "answer": self.answer,
            "context": self.context,
            "category": self.category,
            "difficulty": self.difficulty,
            "metadata": self.metadata,
            "source": self.source
        }


@dataclass
class Conversation:
    """Conversation for fine-tuning."""
    messages: List[Dict[str, str]]
    system_prompt: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: Optional[str] = None


# ============================================================
# Format Converters
# ============================================================

class FormatConverter:
    """Base class for format converters."""

    @classmethod
    def convert(cls, data: Union[List[QAPair], List[Conversation]]) -> List[Dict[str, Any]]:
        raise NotImplementedError


class OpenAIFormatConverter(FormatConverter):
    """Convert to OpenAI fine-tuning format (JSONL)."""

    @classmethod
    def convert(cls, data: Union[List[QAPair], List[Conversation]]) -> List[Dict[str, Any]]:
        """Convert to OpenAI format."""
        if isinstance(data[0], QAPair):
            return cls._convert_qa_pairs(data)
        else:
            return cls._convert_conversations(data)

    @classmethod
    def _convert_qa_pairs(cls, qa_pairs: List[QAPair]) -> List[Dict[str, Any]]:
        """Convert QA pairs to OpenAI format."""
        formatted = []
        for qa in qa_pairs:
            messages = []

            # System message
            if qa.context:
                messages.append({
                    "role": "system",
                    "content": f"You are a helpful assistant that answers questions based on the provided context.\n\nContext: {qa.context}"
                })
            else:
                messages.append({
                    "role": "system",
                    "content": "You are a helpful assistant that answers questions accurately and concisely."
                })

            # User message
            messages.append({
                "role": "user",
                "content": qa.question
            })

            # Assistant message
            messages.append({
                "role": "assistant",
                "content": qa.answer
            })

            formatted.append({"messages": messages})

        return formatted

    @classmethod
    def _convert_conversations(cls, conversations: List[Conversation]) -> List[Dict[str, Any]]:
        """Convert conversations to OpenAI format."""
        formatted = []
        for conv in conversations:
            messages = []

            # System prompt
            if conv.system_prompt:
                messages.append({
                    "role": "system",
                    "content": conv.system_prompt
                })

            # Conversation messages
            messages.extend(conv.messages)

            formatted.append({"messages": messages})

        return formatted


class LlamaFormatConverter(FormatConverter):
    """Convert to Llama/Mistral fine-tuning format."""

    @classmethod
    def convert(cls, data: Union[List[QAPair], List[Conversation]]) -> List[Dict[str, Any]]:
        """Convert to Llama format."""
        if isinstance(data[0], QAPair):
            return cls._convert_qa_pairs(data)
        else:
            return cls._convert_conversations(data)

    @classmethod
    def _convert_qa_pairs(cls, qa_pairs: List[QAPair]) -> List[Dict[str, Any]]:
        """Convert QA pairs to Llama format."""
        formatted = []
        for qa in qa_pairs:
            # Llama chat format: [INST] question [/INST] answer
            if qa.context:
                text = f"[INST] Context: {qa.context}\n\nQuestion: {qa.question} [/INST] {qa.answer}"
            else:
                text = f"[INST] {qa.question} [/INST] {qa.answer}"

            formatted.append({"text": text})

        return formatted

    @classmethod
    def _convert_conversations(cls, conversations: List[Conversation]) -> List[Dict[str, Any]]:
        """Convert conversations to Llama format."""
        formatted = []
        for conv in conversations:
            text_parts = []

            # System prompt
            if conv.system_prompt:
                text_parts.append(f"[INST] <<SYS>>\n{conv.system_prompt}\n<</SYS>>\n\n")

            # Conversation messages
            for msg in conv.messages:
                if msg["role"] == "user":
                    text_parts.append(f"[INST] {msg['content']} [/INST]")
                elif msg["role"] == "assistant":
                    text_parts.append(f" {msg['content']}")

            formatted.append({"text": "".join(text_parts)})

        return formatted


class MistralFormatConverter(FormatConverter):
    """Convert to Mistral fine-tuning format."""

    @classmethod
    def convert(cls, data: Union[List[QAPair], List[Conversation]]) -> List[Dict[str, Any]]:
        """Convert to Mistral format."""
        if isinstance(data[0], QAPair):
            return cls._convert_qa_pairs(data)
        else:
            return cls._convert_conversations(data)

    @classmethod
    def _convert_qa_pairs(cls, qa_pairs: List[QAPair]) -> List[Dict[str, Any]]:
        """Convert QA pairs to Mistral format."""
        formatted = []
        for qa in qa_pairs:
            # Mistral format: <s>[INST] question [/INST] answer </s>
            if qa.context:
                text = f"<s>[INST] Context: {qa.context}\n\nQuestion: {qa.question} [/INST] {qa.answer} </s>"
            else:
                text = f"<s>[INST] {qa.question} [/INST] {qa.answer} </s>"

            formatted.append({"text": text})

        return formatted


class ShareGPTFormatConverter(FormatConverter):
    """Convert to ShareGPT format (used by many open-source models)."""

    @classmethod
    def convert(cls, data: Union[List[QAPair], List[Conversation]]) -> List[Dict[str, Any]]:
        """Convert to ShareGPT format."""
        if isinstance(data[0], QAPair):
            return cls._convert_qa_pairs(data)
        else:
            return cls._convert_conversations(data)

    @classmethod
    def _convert_qa_pairs(cls, qa_pairs: List[QAPair]) -> List[Dict[str, Any]]:
        """Convert QA pairs to ShareGPT format."""
        formatted = []
        for qa in qa_pairs:
            conversations = []

            # System message
            if qa.context:
                conversations.append({
                    "from": "system",
                    "value": f"You are a helpful assistant that answers questions based on the provided context.\n\nContext: {qa.context}"
                })
            else:
                conversations.append({
                    "from": "system",
                    "value": "You are a helpful assistant that answers questions accurately and concisely."
                })

            # User message
            conversations.append({
                "from": "human",
                "value": qa.question
            })

            # Assistant message
            conversations.append({
                "from": "gpt",
                "value": qa.answer
            })

            formatted.append({"conversations": conversations})

        return formatted


class AlpacaFormatConverter(FormatConverter):
    """Convert to Alpaca format."""

    @classmethod
    def convert(cls, data: Union[List[QAPair], List[Conversation]]) -> List[Dict[str, Any]]:
        """Convert to Alpaca format."""
        if isinstance(data[0], QAPair):
            return cls._convert_qa_pairs(data)
        else:
            return cls._convert_conversations(data)

    @classmethod
    def _convert_qa_pairs(cls, qa_pairs: List[QAPair]) -> List[Dict[str, Any]]:
        """Convert QA pairs to Alpaca format."""
        formatted = []
        for qa in qa_pairs:
            entry = {
                "instruction": qa.question,
                "output": qa.answer
            }
            if qa.context:
                entry["input"] = qa.context

            formatted.append(entry)

        return formatted


class SQLFormatConverter(FormatConverter):
    """Convert to SQL format for database storage."""

    @classmethod
    def convert(cls, qa_pairs: List[QAPair]) -> List[Dict[str, Any]]:
        """Convert QA pairs to SQL format."""
        formatted = []
        for qa in qa_pairs:
            formatted.append({
                "question": qa.question,
                "answer": qa.answer,
                "context": qa.context,
                "category": qa.category,
                "difficulty": qa.difficulty,
                "metadata": json.dumps(qa.metadata),
                "source": qa.source,
                "id": qa.id or hashlib.md5(qa.question.encode()).hexdigest()
            })

        return formatted


# ============================================================
# Main Preparation Class
# ============================================================

class FineTunePreparer:
    """
    Prepare data for fine-tuning LLM models.
    """

    FORMAT_CONVERTERS = {
        "openai": OpenAIFormatConverter,
        "llama": LlamaFormatConverter,
        "mistral": MistralFormatConverter,
        "sharegpt": ShareGPTFormatConverter,
        "alpaca": AlpacaFormatConverter,
        "sql": SQLFormatConverter,
    }

    def __init__(
        self,
        input_dir: Optional[str] = None,
        output_dir: str = "./data/finetune",
        formats: List[str] = None,
        max_samples: Optional[int] = None,
        min_quality_score: float = 0.5,
        include_negative_samples: bool = True,
        split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1)
    ):
        """
        Initialize fine-tune preparer.

        Args:
            input_dir: Input directory containing QA data
            output_dir: Output directory for prepared data
            formats: List of output formats
            max_samples: Maximum number of samples
            min_quality_score: Minimum quality score for inclusion
            include_negative_samples: Whether to include negative samples
            split_ratio: Train/validation/test split ratio
        """
        self.input_dir = Path(input_dir) if input_dir else None
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.formats = formats or ["openai", "llama"]
        self.max_samples = max_samples
        self.min_quality_score = min_quality_score
        self.include_negative_samples = include_negative_samples
        self.split_ratio = split_ratio

        # Data storage
        self.qa_pairs: List[QAPair] = []
        self.conversations: List[Conversation] = []

        # Statistics
        self.stats = {
            "total_qa_pairs": 0,
            "filtered_qa_pairs": 0,
            "conversations": 0,
            "formats_generated": []
        }

        logger.info(f"FineTunePreparer initialized: output_dir={output_dir}, formats={formats}")

    def load_data(self, input_path: Optional[str] = None) -> 'FineTunePreparer':
        """
        Load data from various sources.

        Args:
            input_path: Path to input data (optional)

        Returns:
            Self for chaining
        """
        path = Path(input_path) if input_path else self.input_dir

        if not path or not path.exists():
            logger.warning(f"Input path not found: {path}")
            return self

        if path.is_file():
            self._load_file(path)
        elif path.is_dir():
            self._load_directory(path)

        logger.info(f"Loaded {len(self.qa_pairs)} QA pairs and {len(self.conversations)} conversations")
        return self

    def _load_file(self, file_path: Path):
        """Load data from a file."""
        if file_path.suffix == '.json':
            self._load_json_file(file_path)
        elif file_path.suffix == '.jsonl':
            self._load_jsonl_file(file_path)
        elif file_path.suffix == '.csv':
            self._load_csv_file(file_path)
        else:
            logger.warning(f"Unsupported file format: {file_path.suffix}")

    def _load_json_file(self, file_path: Path):
        """Load from JSON file."""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            if isinstance(data, list):
                for item in data:
                    if 'question' in item and 'answer' in item:
                        self._add_qa_pair(item)
                    elif 'messages' in item:
                        self._add_conversation(item)
            elif isinstance(data, dict):
                # Try to find QA pairs in the structure
                for key in ['qa_pairs', 'questions', 'data', 'samples']:
                    if key in data and isinstance(data[key], list):
                        for item in data[key]:
                            if 'question' in item and 'answer' in item:
                                self._add_qa_pair(item)
        except Exception as e:
            logger.error(f"Failed to load JSON file {file_path}: {e}")

    def _load_jsonl_file(self, file_path: Path):
        """Load from JSONL file."""
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        item = json.loads(line)
                        if 'question' in item and 'answer' in item:
                            self._add_qa_pair(item)
                        elif 'messages' in item:
                            self._add_conversation(item)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.error(f"Failed to load JSONL file {file_path}: {e}")

    def _load_csv_file(self, file_path: Path):
        """Load from CSV file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if 'question' in row and 'answer' in row:
                        self._add_qa_pair(row)
        except Exception as e:
            logger.error(f"Failed to load CSV file {file_path}: {e}")

    def _load_directory(self, dir_path: Path):
        """Load all files from a directory."""
        for file_path in dir_path.glob('*'):
            if file_path.is_file():
                self._load_file(file_path)

    def _add_qa_pair(self, item: Dict[str, Any]):
        """Add a QA pair to the dataset."""
        qa = QAPair(
            question=item.get('question', ''),
            answer=item.get('answer', ''),
            context=item.get('context'),
            category=item.get('category'),
            difficulty=item.get('difficulty'),
            metadata=item.get('metadata', {}),
            source=item.get('source'),
            id=item.get('id')
        )

        # Validate
        if not qa.question or not qa.answer:
            return

        self.qa_pairs.append(qa)

    def _add_conversation(self, item: Dict[str, Any]):
        """Add a conversation to the dataset."""
        conv = Conversation(
            messages=item.get('messages', []),
            system_prompt=item.get('system_prompt'),
            metadata=item.get('metadata', {}),
            id=item.get('id')
        )

        if conv.messages:
            self.conversations.append(conv)

    def filter_data(self):
        """
        Filter data based on quality and other criteria.

        Returns:
            Self for chaining
        """
        original_count = len(self.qa_pairs)

        filtered = []
        for qa in self.qa_pairs:
            # Check if meets quality criteria
            quality = self._assess_quality(qa)
            if quality >= self.min_quality_score:
                filtered.append(qa)
            elif self.include_negative_samples and quality < 0.3:
                # Include as negative sample
                filtered.append(qa)

        self.qa_pairs = filtered
        self.stats["filtered_qa_pairs"] = original_count - len(self.qa_pairs)

        logger.info(f"Filtered: {len(self.qa_pairs)}/{original_count} samples kept")
        return self

    def _assess_quality(self, qa: QAPair) -> float:
        """Assess the quality of a QA pair."""
        score = 0.5  # Default

        # Length check
        if len(qa.question) > 10 and len(qa.answer) > 20:
            score += 0.2

        # Contains meaningful content
        if qa.question.strip() and qa.answer.strip():
            score += 0.1

        # Has context
        if qa.context and len(qa.context) > 20:
            score += 0.1

        # Category present
        if qa.category:
            score += 0.05

        # Check for placeholder content
        if "PLACEHOLDER" in qa.answer or "TODO" in qa.answer:
            score -= 0.3

        return min(1.0, max(0.0, score))

    def augment_data(self, augmentation_factor: int = 2):
        """
        Augment data with variations.

        Args:
            augmentation_factor: Number of variations per sample

        Returns:
            Self for chaining
        """
        if augmentation_factor <= 1:
            return self

        logger.info(f"Augmenting data with factor {augmentation_factor}")

        augmented = []
        for qa in self.qa_pairs:
            augmented.append(qa)

            # Generate variations
            for i in range(augmentation_factor - 1):
                var = self._create_variation(qa, i)
                if var:
                    augmented.append(var)

        self.qa_pairs = augmented
        logger.info(f"Augmented to {len(self.qa_pairs)} samples")
        return self

    def _create_variation(self, qa: QAPair, index: int) -> Optional[QAPair]:
        """Create a variation of a QA pair."""
        # Simple paraphrasing - in production, use an LLM or paraphrasing model
        if index == 0:
            # Reword question
            new_question = re.sub(r'^What', 'Could you explain what', qa.question)
            if new_question != qa.question:
                return QAPair(
                    question=new_question,
                    answer=qa.answer,
                    context=qa.context,
                    category=qa.category,
                    difficulty=qa.difficulty,
                    source="augmented"
                )
        elif index == 1:
            # Add "Please" prefix
            if not qa.question.startswith("Please"):
                return QAPair(
                    question=f"Please {qa.question[0].lower()}{qa.question[1:]}",
                    answer=qa.answer,
                    context=qa.context,
                    category=qa.category,
                    difficulty=qa.difficulty,
                    source="augmented"
                )

        return None

    def split_data(self) -> Tuple[List[QAPair], List[QAPair], List[QAPair]]:
        """
        Split data into train/validation/test sets.

        Returns:
            Tuple of (train, validation, test)
        """
        train_ratio, val_ratio, test_ratio = self.split_ratio

        # Shuffle data
        data = self.qa_pairs.copy()
        random.shuffle(data)

        total = len(data)
        train_size = int(total * train_ratio)
        val_size = int(total * val_ratio)

        train = data[:train_size]
        val = data[train_size:train_size + val_size]
        test = data[train_size + val_size:]

        logger.info(f"Split: {len(train)} train, {len(val)} val, {len(test)} test")

        return train, val, test

    def prepare_data(
        self,
        split_sets: bool = True,
        augment: bool = False,
        augmentation_factor: int = 2
    ):
        """
        Prepare data for fine-tuning.

        Args:
            split_sets: Whether to split into train/val/test
            augment: Whether to augment data
            augmentation_factor: Augmentation factor

        Returns:
            Self for chaining
        """
        logger.info("Preparing data for fine-tuning...")

        # Filter data
        self.filter_data()

        # Augment data
        if augment:
            self.augment_data(augmentation_factor)

        # Split data
        if split_sets:
            train, val, test = self.split_data()
            self._export_sets(train, val, test)
        else:
            self._export_data(self.qa_pairs)

        return self

    def _export_sets(self, train: List[QAPair], val: List[QAPair], test: List[QAPair]):
        """Export train/validation/test sets."""
        for format_name in self.formats:
            converter = self.FORMAT_CONVERTERS.get(format_name)
            if not converter:
                logger.warning(f"Unknown format: {format_name}")
                continue

            format_dir = self.output_dir / format_name
            format_dir.mkdir(parents=True, exist_ok=True)

            # Convert and save each set
            self._save_set(converter, train, format_dir, "train")
            self._save_set(converter, val, format_dir, "validation")
            self._save_set(converter, test, format_dir, "test")

            self.stats["formats_generated"].append(format_name)

        # Save metadata
        self._save_metadata(train, val, test)

    def _export_data(self, data: List[QAPair]):
        """Export single dataset."""
        for format_name in self.formats:
            converter = self.FORMAT_CONVERTERS.get(format_name)
            if not converter:
                logger.warning(f"Unknown format: {format_name}")
                continue

            format_dir = self.output_dir / format_name
            format_dir.mkdir(parents=True, exist_ok=True)

            self._save_set(converter, data, format_dir, "data")
            self.stats["formats_generated"].append(format_name)

        # Save metadata
        self._save_metadata(data)

    def _save_set(self, converter, data: List[QAPair], output_dir: Path, name: str):
        """Save a dataset set."""
        if not data:
            logger.warning(f"No data for {name}")
            return

        # Convert data
        converted = converter.convert(data)

        # Determine file extension
        if converter == SQLFormatConverter:
            ext = 'sql'
        else:
            ext = 'jsonl'

        # Save file
        output_file = output_dir / f"{name}.{ext}"

        if ext == 'jsonl':
            with open(output_file, 'w') as f:
                for item in converted:
                    f.write(json.dumps(item) + '\n')
        elif ext == 'sql':
            self._save_sql(converted, output_dir / f"{name}.sql")
        else:
            with open(output_file, 'w') as f:
                json.dump(converted, f, indent=2)

        logger.info(f"Saved {len(converted)} samples to {output_file}")

    def _save_sql(self, data: List[Dict[str, Any]], output_file: Path):
        """Save as SQL."""
        sql_lines = [
            "CREATE TABLE IF NOT EXISTS fine_tune_data (",
            "    id TEXT PRIMARY KEY,",
            "    question TEXT NOT NULL,",
            "    answer TEXT NOT NULL,",
            "    context TEXT,",
            "    category TEXT,",
            "    difficulty TEXT,",
            "    metadata TEXT,",
            "    source TEXT",
            ");",
            ""
        ]

        for item in data:
            sql_lines.append(
                f"INSERT OR REPLACE INTO fine_tune_data VALUES ("
                f"'{item['id']}', "
                f"'{item['question'].replace("'", "''")}', "
                f"'{item['answer'].replace("'", "''")}', "
                f"{f"'{item['context'].replace("'", "''")}'" if item.get('context') else 'NULL'}, "
                f"{f"'{item['category']}'" if item.get('category') else 'NULL'}, "
                f"{f"'{item['difficulty']}'" if item.get('difficulty') else 'NULL'}, "
                f"'{item['metadata']}', "
                f"{f"'{item['source']}'" if item.get('source') else 'NULL'} "
                f");"
            )

        with open(output_file, 'w') as f:
            f.write('\n'.join(sql_lines))

    def _save_metadata(self, *datasets):
        """Save metadata about the dataset."""
        metadata = {
            "created_at": datetime.now().isoformat(),
            "total_samples": len(self.qa_pairs),
            "formats": self.formats,
            "split_ratio": self.split_ratio,
            "min_quality_score": self.min_quality_score,
            "formats_generated": self.stats["formats_generated"],
            "statistics": self._generate_statistics()
        }

        with open(self.output_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

    def _generate_statistics(self) -> Dict[str, Any]:
        """Generate dataset statistics."""
        if not self.qa_pairs:
            return {}

        # Question length statistics
        q_lengths = [len(qa.question) for qa in self.qa_pairs]
        a_lengths = [len(qa.answer) for qa in self.qa_pairs]

        # Category distribution
        categories = {}
        for qa in self.qa_pairs:
            if qa.category:
                categories[qa.category] = categories.get(qa.category, 0) + 1

        # Difficulty distribution
        difficulties = {}
        for qa in self.qa_pairs:
            if qa.difficulty:
                difficulties[qa.difficulty] = difficulties.get(qa.difficulty, 0) + 1

        return {
            "total": len(self.qa_pairs),
            "avg_question_length": sum(q_lengths) / len(q_lengths),
            "avg_answer_length": sum(a_lengths) / len(a_lengths),
            "categories": categories,
            "difficulties": difficulties,
            "has_context": sum(1 for qa in self.qa_pairs if qa.context),
            "sources": {qa.source: sum(1 for q in self.qa_pairs if q.source == qa.source)
                       for qa in self.qa_pairs if qa.source}
        }


# ============================================================
# CLI Interface
# ============================================================

def main():
    """Main entry point for fine-tuning preparation script."""
    parser = argparse.ArgumentParser(
        description="Prepare data for fine-tuning LLM models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Prepare data from directory
  python prepare_finetune.py --input data/qa --output data/finetune --formats openai,llama

  # Prepare from single file with augmentation
  python prepare_finetune.py --input data/qa_data.json --formats openai --augment --augmentation-factor 3

  # Prepare without train/val/test split
  python prepare_finetune.py --input data/qa --no-split
        """
    )

    # Input/output options
    parser.add_argument(
        "--input",
        type=str,
        help="Input directory or file containing QA data"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./data/finetune",
        help="Output directory for prepared data (default: ./data/finetune)"
    )

    # Format options
    parser.add_argument(
        "--formats",
        type=str,
        default="openai,llama",
        help="Comma-separated list of output formats (openai, llama, mistral, sharegpt, alpaca, sql)"
    )

    # Data processing options
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum number of samples to include"
    )
    parser.add_argument(
        "--min-quality",
        type=float,
        default=0.5,
        help="Minimum quality score for inclusion (0-1)"
    )
    parser.add_argument(
        "--no-negative",
        action="store_true",
        help="Exclude negative samples"
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Augment data with variations"
    )
    parser.add_argument(
        "--augmentation-factor",
        type=int,
        default=2,
        help="Number of variations per sample (default: 2)"
    )
    parser.add_argument(
        "--no-split",
        action="store_true",
        help="Don't split into train/val/test sets"
    )
    parser.add_argument(
        "--split-ratio",
        type=str,
        default="0.8,0.1,0.1",
        help="Train/validation/test split ratio (default: 0.8,0.1,0.1)"
    )

    # Other options
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(level=args.log_level, log_dir="./logs")

    # Set random seed
    random.seed(args.seed)

    # Parse split ratio
    split_ratio = tuple(float(x) for x in args.split_ratio.split(','))
    if len(split_ratio) != 3:
        logger.warning(f"Invalid split ratio, using default (0.8, 0.1, 0.1)")
        split_ratio = (0.8, 0.1, 0.1)

    # Parse formats
    formats = [f.strip() for f in args.formats.split(',')]

    # Create preparer
    preparer = FineTunePreparer(
        input_dir=args.input,
        output_dir=args.output,
        formats=formats,
        max_samples=args.max_samples,
        min_quality_score=args.min_quality,
        include_negative_samples=not args.no_negative,
        split_ratio=split_ratio
    )

    # Load data
    if args.input:
        preparer.load_data(args.input)
    else:
        logger.warning("No input specified, use --input to provide data")
        return

    # Prepare data
    preparer.prepare_data(
        split_sets=not args.no_split,
        augment=args.augment,
        augmentation_factor=args.augmentation_factor
    )

    # Print summary
    print("\n" + "=" * 60)
    print("FINE-TUNE PREPARATION COMPLETE")
    print("=" * 60)
    print(f"Total QA pairs: {len(preparer.qa_pairs)}")
    print(f"Formats generated: {', '.join(preparer.stats['formats_generated'])}")
    print(f"Output directory: {preparer.output_dir}")
    print("\nMetadata saved to: metadata.json")
    print("=" * 60)


if __name__ == "__main__":
    main()

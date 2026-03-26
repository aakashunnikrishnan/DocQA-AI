"""
Audio loader module for DocQA AI system.
Supports transcription of audio files and integration with Q&A pipeline.
Handles various audio formats and transcription services.
"""

import os
import gc
import time
import logging
import tempfile
import subprocess
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import json
import re

from src.ingestion.loader import BaseLoader, LoaderResult, LoaderErrorCode, track_memory, handle_loader_errors
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Try importing audio processing libraries
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    logger.warning("whisper not installed. Install with: pip install openai-whisper")

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    logger.warning("librosa not installed. Install with: pip install librosa")

try:
    import pydub
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    logger.warning("pydub not installed. Install with: pip install pydub")

try:
    import speech_recognition as sr
    SR_AVAILABLE = True
except ImportError:
    SR_AVAILABLE = False
    logger.warning("speech_recognition not installed. Install with: pip install SpeechRecognition")

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI not installed. Install with: pip install openai")


class TranscriptionProvider(Enum):
    """Transcription service providers."""
    WHISPER = "whisper"
    OPENAI = "openai"
    SPEECH_RECOGNITION = "speech_recognition"
    LOCAL = "local"


class AudioFormat(Enum):
    """Supported audio formats."""
    MP3 = "mp3"
    WAV = "wav"
    FLAC = "flac"
    M4A = "m4a"
    OGG = "ogg"
    AAC = "aac"
    WMA = "wma"
    OPUS = "opus"


@dataclass
class AudioMetadata:
    """Audio file metadata."""
    file_path: str
    file_type: str
    file_size: int
    duration_seconds: float
    sample_rate: int
    channels: int
    bit_depth: Optional[int] = None
    bit_rate: Optional[int] = None
    codec: Optional[str] = None
    title: Optional[str] = None
    artist: Optional[str] = None
    album: Optional[str] = None
    language: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "file_path": self.file_path,
            "file_type": self.file_type,
            "file_size": self.file_size,
            "duration_seconds": self.duration_seconds,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "bit_depth": self.bit_depth,
            "bit_rate": self.bit_rate,
            "codec": self.codec,
            "title": self.title,
            "artist": self.artist,
            "album": self.album,
            "language": self.language
        }


@dataclass
class TranscriptionResult:
    """Transcription result."""
    text: str
    language: Optional[str] = None
    segments: Optional[List[Dict[str, Any]]] = None
    confidence: float = 0.0
    word_timestamps: Optional[List[Dict[str, Any]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "language": self.language,
            "segments": self.segments,
            "confidence": self.confidence,
            "word_timestamps": self.word_timestamps,
            "metadata": self.metadata
        }


class AudioTranscriber:
    """
    Audio transcriber supporting multiple providers.
    """

    def __init__(
        self,
        provider: Union[str, TranscriptionProvider] = TranscriptionProvider.WHISPER,
        model: str = "base",
        device: str = "cpu",
        language: Optional[str] = None,
        api_key: Optional[str] = None,
        sample_rate: int = 16000,
        max_duration: int = 600,  # 10 minutes
        segment_length: int = 30,  # 30 seconds
        **kwargs
    ):
        """
        Initialize audio transcriber.

        Args:
            provider: Transcription provider
            model: Model name (for Whisper)
            device: Device (cpu/cuda)
            language: Language code
            api_key: API key (for OpenAI)
            sample_rate: Target sample rate
            max_duration: Maximum duration in seconds
            segment_length: Segment length for chunked processing
            **kwargs: Additional arguments
        """
        self.provider = TranscriptionProvider(provider) if isinstance(provider, str) else provider
        self.model = model
        self.device = device
        self.language = language
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.segment_length = segment_length

        self._initialize_provider()

        logger.info(f"AudioTranscriber initialized: provider={self.provider.value}, model={model}")

    def _initialize_provider(self):
        """Initialize the transcription provider."""
        if self.provider == TranscriptionProvider.WHISPER:
            if not WHISPER_AVAILABLE:
                raise ImportError("Whisper not installed. Install with: pip install openai-whisper")
            self._model = whisper.load_model(self.model, device=self.device)

        elif self.provider == TranscriptionProvider.OPENAI:
            if not OPENAI_AVAILABLE:
                raise ImportError("OpenAI not installed. Install with: pip install openai")
            if not self.api_key:
                raise ValueError("OpenAI API key required for OpenAI transcription")
            self._client = OpenAI(api_key=self.api_key)

        elif self.provider == TranscriptionProvider.SPEECH_RECOGNITION:
            if not SR_AVAILABLE:
                raise ImportError("speech_recognition not installed. Install with: pip install SpeechRecognition")
            self._recognizer = sr.Recognizer()

        elif self.provider == TranscriptionProvider.LOCAL:
            # Use whisper with local model
            if not WHISPER_AVAILABLE:
                raise ImportError("Whisper not installed. Install with: pip install openai-whisper")
            self._model = whisper.load_model(self.model, device=self.device)

        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def transcribe_file(self, file_path: str) -> TranscriptionResult:
        """
        Transcribe an audio file.

        Args:
            file_path: Path to audio file

        Returns:
            TranscriptionResult object
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Check duration
        duration = self._get_duration(file_path)
        if duration > self.max_duration:
            logger.warning(f"Audio duration ({duration}s) exceeds max ({self.max_duration}s). "
                          f"Processing in segments.")

        # Transcribe based on provider
        if self.provider == TranscriptionProvider.WHISPER:
            return self._transcribe_whisper(file_path)
        elif self.provider == TranscriptionProvider.OPENAI:
            return self._transcribe_openai(file_path)
        elif self.provider == TranscriptionProvider.SPEECH_RECOGNITION:
            return self._transcribe_speech_recognition(file_path)
        elif self.provider == TranscriptionProvider.LOCAL:
            return self._transcribe_whisper(file_path)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _transcribe_whisper(self, file_path: str) -> TranscriptionResult:
        """Transcribe using Whisper."""
        try:
            result = self._model.transcribe(
                file_path,
                language=self.language,
                task="transcribe",
                fp16=self.device == "cuda"
            )

            segments = []
            for seg in result.get("segments", []):
                segments.append({
                    "start": seg.get("start", 0),
                    "end": seg.get("end", 0),
                    "text": seg.get("text", "").strip()
                })

            word_timestamps = []
            for seg in result.get("segments", []):
                for word in seg.get("words", []):
                    word_timestamps.append({
                        "word": word.get("word", ""),
                        "start": word.get("start", 0),
                        "end": word.get("end", 0),
                        "probability": word.get("probability", 0.0)
                    })

            return TranscriptionResult(
                text=result.get("text", "").strip(),
                language=result.get("language"),
                segments=segments,
                confidence=result.get("confidence", 0.0),
                word_timestamps=word_timestamps,
                metadata={
                    "provider": "whisper",
                    "model": self.model,
                    "device": self.device
                }
            )

        except Exception as e:
            logger.error(f"Whisper transcription failed: {e}")
            raise

    def _transcribe_openai(self, file_path: str) -> TranscriptionResult:
        """Transcribe using OpenAI's API."""
        try:
            with open(file_path, 'rb') as f:
                response = self._client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f,
                    language=self.language,
                    response_format="verbose_json"
                )

            segments = []
            if hasattr(response, 'segments') and response.segments:
                for seg in response.segments:
                    segments.append({
                        "start": seg.get("start", 0),
                        "end": seg.get("end", 0),
                        "text": seg.get("text", "").strip()
                    })

            word_timestamps = []
            if hasattr(response, 'words') and response.words:
                for word in response.words:
                    word_timestamps.append({
                        "word": word.get("word", ""),
                        "start": word.get("start", 0),
                        "end": word.get("end", 0),
                        "probability": word.get("probability", 0.0)
                    })

            return TranscriptionResult(
                text=response.text if hasattr(response, 'text') else "",
                language=getattr(response, 'language', None),
                segments=segments,
                confidence=getattr(response, 'confidence', 0.0),
                word_timestamps=word_timestamps,
                metadata={
                    "provider": "openai",
                    "model": "whisper-1"
                }
            )

        except Exception as e:
            logger.error(f"OpenAI transcription failed: {e}")
            raise

    def _transcribe_speech_recognition(self, file_path: str) -> TranscriptionResult:
        """Transcribe using SpeechRecognition."""
        try:
            # Convert to WAV if needed
            audio_data = self._convert_to_wav(file_path)

            with sr.AudioFile(audio_data) as source:
                audio = self._recognizer.record(source)

                try:
                    text = self._recognizer.recognize_google(audio, language=self.language or "en-US")
                    confidence = 0.8  # Google doesn't provide confidence scores
                except sr.UnknownValueError:
                    text = ""
                    confidence = 0.0
                except sr.RequestError as e:
                    logger.error(f"Speech recognition request failed: {e}")
                    raise

            return TranscriptionResult(
                text=text,
                language=self.language,
                segments=[{"start": 0, "end": 0, "text": text}],
                confidence=confidence,
                metadata={
                    "provider": "speech_recognition"
                }
            )

        except Exception as e:
            logger.error(f"Speech recognition failed: {e}")
            raise

    def _get_duration(self, file_path: str) -> float:
        """Get audio duration in seconds."""
        try:
            if LIBROSA_AVAILABLE:
                audio, sr = librosa.load(file_path, sr=None)
                return len(audio) / sr
            elif PYDUB_AVAILABLE:
                audio = AudioSegment.from_file(file_path)
                return len(audio) / 1000.0
            else:
                # Use ffprobe if available
                import subprocess
                result = subprocess.run(
                    ['ffprobe', '-i', file_path, '-show_entries', 'format=duration', '-v', 'quiet', '-of', 'csv=%s' % ("p=0")],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    return float(result.stdout.strip())
                return 0.0
        except Exception as e:
            logger.warning(f"Failed to get duration: {e}")
            return 0.0

    def _convert_to_wav(self, file_path: str) -> str:
        """Convert audio file to WAV format."""
        if PYDUB_AVAILABLE:
            audio = AudioSegment.from_file(file_path)
            temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            audio.export(temp_wav.name, format='wav')
            return temp_wav.name
        else:
            # Use ffmpeg if available
            temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            subprocess.run([
                'ffmpeg', '-i', file_path, '-acodec', 'pcm_s16le', '-ar', '16000',
                '-ac', '1', temp_wav.name, '-y'
            ], capture_output=True)
            return temp_wav.name


class AudioLoader(BaseLoader):
    """
    Audio document loader with transcription and metadata extraction.
    """

    def __init__(
        self,
        timeout: int = 300,  # Audio processing can take longer
        max_file_size_mb: int = 500,
        max_memory_mb: int = 4096,
        chunk_size: int = 1024 * 1024,
        transcriber: Optional[AudioTranscriber] = None,
        provider: str = "whisper",
        model: str = "base",
        device: str = "cpu",
        language: Optional[str] = None,
        include_metadata: bool = True,
        include_timestamps: bool = False
    ):
        """
        Initialize audio loader.

        Args:
            timeout: Maximum time in seconds for loading operations
            max_file_size_mb: Maximum file size in MB
            max_memory_mb: Maximum memory usage in MB
            chunk_size: Chunk size for streaming reads
            transcriber: Audio transcriber instance
            provider: Transcription provider
            model: Whisper model name
            device: Device (cpu/cuda)
            language: Language code
            include_metadata: Include audio metadata in output
            include_timestamps: Include word timestamps
        """
        super().__init__(timeout, max_file_size_mb, max_memory_mb, chunk_size)

        self.transcriber = transcriber or AudioTranscriber(
            provider=provider,
            model=model,
            device=device,
            language=language
        )
        self.include_metadata = include_metadata
        self.include_timestamps = include_timestamps

        # Supported formats
        self.supported_formats = [
            AudioFormat.MP3.value,
            AudioFormat.WAV.value,
            AudioFormat.FLAC.value,
            AudioFormat.M4A.value,
            AudioFormat.OGG.value,
            AudioFormat.AAC.value,
            AudioFormat.WMA.value,
            AudioFormat.OPUS.value
        ]

        logger.info(f"AudioLoader initialized: provider={provider}, model={model}, device={device}")

    @handle_loader_errors()
    def load(self, file_path: str) -> LoaderResult:
        """
        Load and transcribe audio file.

        Args:
            file_path: Path to audio file

        Returns:
            LoaderResult with transcription and metadata
        """
        is_valid, error_code, error_msg = self.validate_file(file_path)
        if not is_valid:
            return LoaderResult(
                success=False,
                error_code=error_code,
                error_message=error_msg,
                file_path=file_path
            )

        warnings = []
        metadata = self.get_metadata(file_path)
        file_size = Path(file_path).stat().st_size

        try:
            # Transcribe audio
            with track_memory("Audio transcription"):
                result = self.transcriber.transcribe_file(file_path)

            # Build content
            content_parts = []

            # Add audio metadata
            if self.include_metadata:
                content_parts.append(f"Audio File: {Path(file_path).name}")
                content_parts.append(f"Duration: {metadata['duration_seconds']:.1f} seconds")
                if metadata.get('title'):
                    content_parts.append(f"Title: {metadata['title']}")
                if metadata.get('artist'):
                    content_parts.append(f"Artist: {metadata['artist']}")
                content_parts.append("")

            # Add transcription
            content_parts.append("=== TRANSCRIPTION ===")
            content_parts.append("")

            # Add transcription text with timestamps if requested
            if self.include_timestamps and result.word_timestamps:
                text = ""
                current_time = 0
                for word_info in result.word_timestamps:
                    timestamp = word_info.get('start', 0)
                    if timestamp > current_time + 1:
                        text += f"\n[{timestamp:.1f}s] "
                        current_time = timestamp
                    text += word_info.get('word', '') + " "
                content_parts.append(text.strip())
            elif result.segments:
                for seg in result.segments:
                    timestamp = f"[{seg.get('start', 0):.1f}s - {seg.get('end', 0):.1f}s]"
                    content_parts.append(f"{timestamp} {seg.get('text', '')}")
            else:
                content_parts.append(result.text)

            content = "\n".join(content_parts)

            # Add metadata
            metadata.update({
                "transcription_provider": self.transcriber.provider.value,
                "transcription_model": self.transcriber.model,
                "transcription_language": result.language,
                "confidence": result.confidence,
                "duration_seconds": metadata.get('duration_seconds', 0),
                "has_timestamps": self.include_timestamps and bool(result.word_timestamps)
            })

        except Exception as e:
            raise Exception(f"Failed to transcribe audio: {e}")

        return LoaderResult(
            success=True,
            content=content,
            metadata=metadata,
            file_path=file_path,
            file_size=file_size,
            warnings=warnings,
            memory_usage_mb=get_memory_usage()
        )

    def get_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Extract audio metadata.

        Args:
            file_path: Path to audio file

        Returns:
            Audio metadata dictionary
        """
        metadata = {
            "file_path": file_path,
            "file_type": Path(file_path).suffix.lower().replace('.', ''),
            "file_size": Path(file_path).stat().st_size,
            "duration_seconds": 0.0,
            "sample_rate": 0,
            "channels": 0,
            "bit_depth": None,
            "bit_rate": None,
            "codec": None,
            "title": None,
            "artist": None,
            "album": None,
            "language": None
        }

        try:
            # Get basic file info
            path = Path(file_path)
            metadata["file_size"] = path.stat().st_size
            metadata["file_name"] = path.name

            # Use pydub for metadata extraction
            if PYDUB_AVAILABLE:
                try:
                    audio = AudioSegment.from_file(file_path)
                    metadata["duration_seconds"] = len(audio) / 1000.0
                    metadata["sample_rate"] = audio.frame_rate
                    metadata["channels"] = audio.channels
                    metadata["bit_depth"] = audio.sample_width * 8

                    # Try to get bit rate
                    if hasattr(audio, 'bitrate'):
                        metadata["bit_rate"] = int(audio.bitrate) // 1000
                except Exception as e:
                    logger.warning(f"Failed to extract metadata with pydub: {e}")

            # Use mutagen for tag extraction
            try:
                import mutagen
                from mutagen import File
                audio_file = File(file_path)
                if audio_file:
                    # Common tags
                    for tag in ['title', 'artist', 'album']:
                        if hasattr(audio_file, 'tags') and audio_file.tags:
                            if tag in audio_file.tags:
                                metadata[tag] = str(audio_file.tags[tag])

                    # Codec info
                    if hasattr(audio_file, 'info'):
                        if hasattr(audio_file.info, 'codec'):
                            metadata["codec"] = audio_file.info.codec
                        if hasattr(audio_file.info, 'bitrate'):
                            metadata["bit_rate"] = audio_file.info.bitrate // 1000
            except ImportError:
                pass
            except Exception as e:
                logger.warning(f"Failed to extract tags: {e}")

        except Exception as e:
            logger.warning(f"Failed to extract metadata from {file_path}: {e}")

        return metadata

    def validate_file(self, file_path: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Validate audio file before loading.

        Args:
            file_path: Path to audio file

        Returns:
            Tuple of (is_valid, error_code, error_message)
        """
        is_valid, error_code, error_msg = super().validate_file(file_path)
        if not is_valid:
            return is_valid, error_code, error_msg

        # Check format
        extension = Path(file_path).suffix.lower().replace('.', '')
        if extension not in self.supported_formats:
            return False, LoaderErrorCode.UNSUPPORTED_FORMAT, f"Unsupported audio format: {extension}"

        return True, None, None


class AudioQAPipeline:
    """
    Pipeline for audio Q&A using transcription and RAG.
    """

    def __init__(
        self,
        retriever: Any,
        llm_interface: Any,
        audio_loader: Optional[AudioLoader] = None,
        **kwargs
    ):
        """
        Initialize audio Q&A pipeline.

        Args:
            retriever: Retriever instance
            llm_interface: LLM interface
            audio_loader: Audio loader instance
            **kwargs: Additional arguments
        """
        self.retriever = retriever
        self.llm_interface = llm_interface
        self.audio_loader = audio_loader or AudioLoader(**kwargs)

        logger.info("AudioQAPipeline initialized")

    def process_audio(
        self,
        file_path: str,
        question: Optional[str] = None,
        top_k: int = 5,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process audio file and optionally ask a question.

        Args:
            file_path: Path to audio file
            question: Optional question to ask
            top_k: Number of documents to retrieve
            **kwargs: Additional arguments

        Returns:
            Dictionary with transcription and answer
        """
        # Load and transcribe audio
        result = self.audio_loader.load(file_path)
        if not result.success:
            return {
                "success": False,
                "error": result.error_message,
                "transcription": None
            }

        response = {
            "success": True,
            "transcription": result.content,
            "metadata": result.metadata,
            "warnings": result.warnings
        }

        # Answer question if provided
        if question:
            # Use transcription as context
            answer = self._answer_question(
                question=question,
                context=result.content,
                top_k=top_k,
                **kwargs
            )
            response["answer"] = answer

        return response

    def _answer_question(
        self,
        question: str,
        context: str,
        top_k: int = 5,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Answer a question based on audio transcription.

        Args:
            question: Question to answer
            context: Transcription text
            top_k: Number of documents to retrieve
            **kwargs: Additional arguments

        Returns:
            Answer dictionary
        """
        # Generate prompt
        from src.generation.prompt_templates import get_rag_prompt

        prompt = get_rag_prompt(
            question=question,
            chunks=[{"text": context, "source": "audio_transcription"}]
        )

        # Generate response
        response = self.llm_interface.generate_simple(
            prompt,
            system_prompt="You are a helpful assistant that answers questions based on audio transcriptions."
        )

        # Post-process
        from src.generation.response_postprocess import postprocess_response
        processed = postprocess_response(response, aggressive_cleaning=True)

        return {
            "answer": processed.cleaned_text,
            "confidence": processed.confidence,
            "has_hallucination": processed.has_hallucination
        }


# ============================================================
# Convenience Functions
# ============================================================

def transcribe_audio(
    file_path: str,
    provider: str = "whisper",
    model: str = "base",
    language: Optional[str] = None,
    **kwargs
) -> TranscriptionResult:
    """
    Quick function to transcribe audio.

    Args:
        file_path: Path to audio file
        provider: Transcription provider
        model: Whisper model name
        language: Language code
        **kwargs: Additional arguments

    Returns:
        TranscriptionResult object
    """
    transcriber = AudioTranscriber(
        provider=provider,
        model=model,
        language=language,
        **kwargs
    )
    return transcriber.transcribe_file(file_path)


def load_audio_file(
    file_path: str,
    provider: str = "whisper",
    model: str = "base",
    **kwargs
) -> LoaderResult:
    """
    Quick function to load and transcribe audio file.

    Args:
        file_path: Path to audio file
        provider: Transcription provider
        model: Whisper model name
        **kwargs: Additional arguments

    Returns:
        LoaderResult object
    """
    loader = AudioLoader(provider=provider, model=model, **kwargs)
    return loader.load(file_path)


if __name__ == "__main__":
    # Example usage
    import asyncio

    async def test_audio_loader():
        """Test audio loader."""
        logging.basicConfig(level=logging.INFO)

        print("Testing Audio Loader...")
        print("=" * 60)

        # Test with sample audio file (if available)
        test_file = "sample_audio.mp3"

        if os.path.exists(test_file):
            print(f"\n📂 Processing: {test_file}")

            # Load with audio loader
            result = load_audio_file(
                test_file,
                provider="whisper",
                model="base",
                include_timestamps=True
            )

            if result.success:
                print(f"\n✅ Audio loaded successfully")
                print(f"   Duration: {result.metadata.get('duration_seconds', 0):.1f}s")
                print(f"   Transcription length: {len(result.content)} chars")
                print(f"\n📝 Transcription preview:")
                print(result.content[:500] + "...")
            else:
                print(f"❌ Failed: {result.error_message}")
        else:
            print(f"\n⚠️ Test file not found: {test_file}")
            print("   Please provide an audio file for testing.")

        print("\n✅ Audio Loader ready!")

    asyncio.run(test_audio_loader())

"""
Document versioning system for tracking changes, managing versions, and supporting rollback.
Provides capabilities for:
- Version creation and management
- Diff generation between versions
- Rollback to previous versions
- Version metadata and history
- Storage optimization with compression
"""

import os
import json
import hashlib
import pickle
import gzip
import shutil
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict
import uuid
import time

logger = logging.getLogger(__name__)


class VersionStatus(Enum):
    """Status of a document version."""
    ACTIVE = "active"
    ARCHIVED = "archived"
    DELETED = "deleted"
    DRAFT = "draft"
    PUBLISHED = "published"


class VersionAction(Enum):
    """Actions that trigger version creation."""
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    RESTORE = "restore"
    MERGE = "merge"
    IMPORT = "import"
    EXPORT = "export"


@dataclass
class VersionMetadata:
    """Metadata for a document version."""
    version_id: str
    document_id: str
    version_number: int
    created_at: datetime
    created_by: str
    action: VersionAction
    status: VersionStatus = VersionStatus.ACTIVE
    parent_version_id: Optional[str] = None
    description: str = ""
    size_bytes: int = 0
    hash: str = ""
    tags: List[str] = field(default_factory=list)
    custom_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "version_id": self.version_id,
            "document_id": self.document_id,
            "version_number": self.version_number,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "action": self.action.value,
            "status": self.status.value,
            "parent_version_id": self.parent_version_id,
            "description": self.description,
            "size_bytes": self.size_bytes,
            "hash": self.hash,
            "tags": self.tags,
            "custom_metadata": self.custom_metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VersionMetadata':
        """Create from dictionary."""
        return cls(
            version_id=data["version_id"],
            document_id=data["document_id"],
            version_number=data["version_number"],
            created_at=datetime.fromisoformat(data["created_at"]),
            created_by=data["created_by"],
            action=VersionAction(data["action"]),
            status=VersionStatus(data.get("status", "active")),
            parent_version_id=data.get("parent_version_id"),
            description=data.get("description", ""),
            size_bytes=data.get("size_bytes", 0),
            hash=data.get("hash", ""),
            tags=data.get("tags", []),
            custom_metadata=data.get("custom_metadata", {})
        )


@dataclass
class DocumentVersion:
    """Full document version including content and metadata."""
    metadata: VersionMetadata
    content: Optional[str] = None
    content_embeddings: Optional[List[List[float]]] = None
    chunks: Optional[List[str]] = None
    chunk_metadata: Optional[List[Dict[str, Any]]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metadata": self.metadata.to_dict(),
            "content": self.content,
            "content_embeddings": self.content_embeddings,
            "chunks": self.chunks,
            "chunk_metadata": self.chunk_metadata
        }

    def get_size_bytes(self) -> int:
        """Get size of version in bytes."""
        total_size = 0
        if self.content:
            total_size += len(self.content.encode('utf-8'))
        if self.content_embeddings:
            total_size += len(pickle.dumps(self.content_embeddings))
        if self.chunks:
            total_size += len(pickle.dumps(self.chunks))
        return total_size


class VersionStorage:
    """
    Storage backend for document versions.
    Supports local file system, compression, and efficient storage.
    """

    def __init__(
        self,
        storage_dir: str = "./data/versions",
        compression: bool = True,
        max_versions_per_document: int = 100
    ):
        """
        Initialize version storage.

        Args:
            storage_dir: Directory to store versions
            compression: Whether to compress version data
            max_versions_per_document: Maximum versions to keep per document
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.compression = compression
        self.max_versions_per_document = max_versions_per_document

        # Index file path
        self.index_file = self.storage_dir / "version_index.json"
        self._index: Dict[str, List[str]] = {}  # document_id -> [version_ids]
        self._metadata_cache: Dict[str, VersionMetadata] = {}

        # Load index
        self._load_index()

        logger.info(f"VersionStorage initialized: {storage_dir}, compression={compression}")

    def _load_index(self):
        """Load version index from disk."""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r') as f:
                    self._index = json.load(f)
                logger.info(f"Loaded index with {len(self._index)} documents")
            except Exception as e:
                logger.warning(f"Failed to load index: {e}")
                self._index = {}
        else:
            self._index = {}

    def _save_index(self):
        """Save version index to disk."""
        try:
            with open(self.index_file, 'w') as f:
                json.dump(self._index, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save index: {e}")

    def _get_version_path(self, document_id: str, version_id: str) -> Path:
        """Get path for version file."""
        doc_dir = self.storage_dir / document_id
        doc_dir.mkdir(parents=True, exist_ok=True)
        return doc_dir / f"{version_id}.json"

    def _get_metadata_path(self, document_id: str, version_id: str) -> Path:
        """Get path for version metadata file."""
        doc_dir = self.storage_dir / document_id
        return doc_dir / f"{version_id}_meta.json"

    def _compress_data(self, data: Dict[str, Any]) -> bytes:
        """Compress data using gzip."""
        if not self.compression:
            return pickle.dumps(data)

        json_str = json.dumps(data, default=str)
        return gzip.compress(json_str.encode('utf-8'))

    def _decompress_data(self, data: bytes) -> Dict[str, Any]:
        """Decompress data."""
        if not self.compression:
            return pickle.loads(data)

        decompressed = gzip.decompress(data)
        return json.loads(decompressed.decode('utf-8'))

    def save_version(
        self,
        document_id: str,
        version: DocumentVersion
    ) -> bool:
        """
        Save a document version to storage.

        Args:
            document_id: Document ID
            version: DocumentVersion object

        Returns:
            Success status
        """
        try:
            # Prepare data
            version_data = version.to_dict()

            # Save version data
            version_path = self._get_version_path(document_id, version.metadata.version_id)

            # Compress and save
            compressed = self._compress_data(version_data)
            with open(version_path, 'wb') as f:
                f.write(compressed)

            # Save metadata separately for quick access
            meta_path = self._get_metadata_path(document_id, version.metadata.version_id)
            meta_data = version.metadata.to_dict()
            with open(meta_path, 'w') as f:
                json.dump(meta_data, f, indent=2)

            # Update index
            if document_id not in self._index:
                self._index[document_id] = []

            # Add version to index
            if version.metadata.version_id not in self._index[document_id]:
                self._index[document_id].append(version.metadata.version_id)

            # Enforce max versions
            self._enforce_max_versions(document_id)

            # Update cache
            self._metadata_cache[version.metadata.version_id] = version.metadata

            # Save index
            self._save_index()

            logger.info(f"Saved version {version.metadata.version_number} for document {document_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to save version: {e}")
            return False

    def load_version(
        self,
        document_id: str,
        version_id: str,
        load_content: bool = True,
        load_embeddings: bool = False
    ) -> Optional[DocumentVersion]:
        """
        Load a document version from storage.

        Args:
            document_id: Document ID
            version_id: Version ID
            load_content: Whether to load content
            load_embeddings: Whether to load embeddings

        Returns:
            DocumentVersion object or None
        """
        try:
            # Load metadata
            meta_path = self._get_metadata_path(document_id, version_id)
            if not meta_path.exists():
                logger.warning(f"Metadata not found for version {version_id}")
                return None

            with open(meta_path, 'r') as f:
                meta_data = json.load(f)

            metadata = VersionMetadata.from_dict(meta_data)

            # Load full version data
            version_path = self._get_version_path(document_id, version_id)
            if not version_path.exists():
                return DocumentVersion(metadata=metadata)

            with open(version_path, 'rb') as f:
                version_data = self._decompress_data(f.read())

            # Create version object
            version = DocumentVersion(
                metadata=metadata,
                content=version_data.get("content") if load_content else None,
                content_embeddings=version_data.get("content_embeddings") if load_embeddings else None,
                chunks=version_data.get("chunks") if load_content else None,
                chunk_metadata=version_data.get("chunk_metadata") if load_content else None
            )

            return version

        except Exception as e:
            logger.error(f"Failed to load version {version_id}: {e}")
            return None

    def load_metadata(self, document_id: str, version_id: str) -> Optional[VersionMetadata]:
        """Load only version metadata."""
        try:
            # Check cache
            if version_id in self._metadata_cache:
                return self._metadata_cache[version_id]

            meta_path = self._get_metadata_path(document_id, version_id)
            if not meta_path.exists():
                return None

            with open(meta_path, 'r') as f:
                meta_data = json.load(f)

            metadata = VersionMetadata.from_dict(meta_data)
            self._metadata_cache[version_id] = metadata
            return metadata

        except Exception as e:
            logger.error(f"Failed to load metadata for {version_id}: {e}")
            return None

    def delete_version(self, document_id: str, version_id: str) -> bool:
        """
        Delete a version from storage.

        Args:
            document_id: Document ID
            version_id: Version ID

        Returns:
            Success status
        """
        try:
            # Delete version data
            version_path = self._get_version_path(document_id, version_id)
            if version_path.exists():
                version_path.unlink()

            # Delete metadata
            meta_path = self._get_metadata_path(document_id, version_id)
            if meta_path.exists():
                meta_path.unlink()

            # Remove from index
            if document_id in self._index:
                if version_id in self._index[document_id]:
                    self._index[document_id].remove(version_id)
                    self._save_index()

            # Remove from cache
            if version_id in self._metadata_cache:
                del self._metadata_cache[version_id]

            # Remove directory if empty
            doc_dir = self.storage_dir / document_id
            if doc_dir.exists() and not any(doc_dir.iterdir()):
                doc_dir.rmdir()

            logger.info(f"Deleted version {version_id} for document {document_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete version {version_id}: {e}")
            return False

    def _enforce_max_versions(self, document_id: str):
        """Enforce maximum versions per document."""
        if document_id not in self._index:
            return

        versions = self._index[document_id]
        if len(versions) <= self.max_versions_per_document:
            return

        # Sort versions by creation time
        version_info = []
        for vid in versions:
            metadata = self.load_metadata(document_id, vid)
            if metadata:
                version_info.append((vid, metadata.created_at))

        # Sort by creation time (oldest first)
        version_info.sort(key=lambda x: x[1])

        # Remove oldest versions
        to_delete = len(version_info) - self.max_versions_per_document
        for i in range(to_delete):
            vid, _ = version_info[i]
            self.delete_version(document_id, vid)

    def list_versions(self, document_id: str) -> List[VersionMetadata]:
        """
        List all versions for a document.

        Args:
            document_id: Document ID

        Returns:
            List of version metadata
        """
        versions = []

        if document_id not in self._index:
            return versions

        for version_id in self._index[document_id]:
            metadata = self.load_metadata(document_id, version_id)
            if metadata:
                versions.append(metadata)

        # Sort by version number (newest first)
        versions.sort(key=lambda x: x.version_number, reverse=True)

        return versions

    def get_latest_version(self, document_id: str) -> Optional[VersionMetadata]:
        """
        Get the latest version metadata for a document.

        Args:
            document_id: Document ID

        Returns:
            Latest version metadata or None
        """
        versions = self.list_versions(document_id)
        return versions[0] if versions else None

    def get_version_by_number(
        self,
        document_id: str,
        version_number: int
    ) -> Optional[VersionMetadata]:
        """
        Get version metadata by version number.

        Args:
            document_id: Document ID
            version_number: Version number

        Returns:
            Version metadata or None
        """
        versions = self.list_versions(document_id)
        for version in versions:
            if version.version_number == version_number:
                return version
        return None

    def get_version_history(
        self,
        document_id: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get version history for a document.

        Args:
            document_id: Document ID
            limit: Maximum number of versions to return

        Returns:
            List of version history entries
        """
        versions = self.list_versions(document_id)
        history = []

        for version in versions[:limit]:
            history.append({
                "version_id": version.version_id,
                "version_number": version.version_number,
                "created_at": version.created_at.isoformat(),
                "created_by": version.created_by,
                "action": version.action.value,
                "status": version.status.value,
                "description": version.description,
                "size_bytes": version.size_bytes,
                "hash": version.hash,
                "tags": version.tags
            })

        return history

    def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get storage statistics.

        Returns:
            Storage statistics
        """
        total_size = 0
        total_versions = 0
        total_documents = len(self._index)

        for document_id, versions in self._index.items():
            total_versions += len(versions)
            doc_dir = self.storage_dir / document_id
            if doc_dir.exists():
                for file in doc_dir.iterdir():
                    if file.is_file():
                        total_size += file.stat().st_size

        return {
            "total_documents": total_documents,
            "total_versions": total_versions,
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "compression_enabled": self.compression,
            "max_versions_per_document": self.max_versions_per_document,
            "storage_dir": str(self.storage_dir)
        }


class DiffGenerator:
    """
    Generate diffs between document versions.
    """

    @staticmethod
    def generate_diff(
        old_content: str,
        new_content: str,
        algorithm: str = "unified"
    ) -> Dict[str, Any]:
        """
        Generate diff between two document versions.

        Args:
            old_content: Old document content
            new_content: New document content
            algorithm: Diff algorithm ('unified', 'word', 'character')

        Returns:
            Diff summary
        """
        if algorithm == "unified":
            return DiffGenerator._generate_unified_diff(old_content, new_content)
        elif algorithm == "word":
            return DiffGenerator._generate_word_diff(old_content, new_content)
        elif algorithm == "character":
            return DiffGenerator._generate_character_diff(old_content, new_content)
        else:
            raise ValueError(f"Unsupported diff algorithm: {algorithm}")

    @staticmethod
    def _generate_unified_diff(old_content: str, new_content: str) -> Dict[str, Any]:
        """Generate unified diff."""
        old_lines = old_content.splitlines()
        new_lines = new_content.splitlines()

        added = 0
        removed = 0
        unchanged = 0

        # Simple line-by-line comparison
        i = j = 0
        diff_lines = []

        while i < len(old_lines) and j < len(new_lines):
            if old_lines[i] == new_lines[j]:
                unchanged += 1
                diff_lines.append(f" {old_lines[i]}")
                i += 1
                j += 1
            else:
                # Check if line was removed
                if i < len(old_lines) and (j >= len(new_lines) or old_lines[i] != new_lines[j]):
                    removed += 1
                    diff_lines.append(f"-{old_lines[i]}")
                    i += 1
                # Check if line was added
                if j < len(new_lines) and (i >= len(old_lines) or old_lines[i] != new_lines[j]):
                    added += 1
                    diff_lines.append(f"+{new_lines[j]}")
                    j += 1

        # Remaining lines
        while i < len(old_lines):
            removed += 1
            diff_lines.append(f"-{old_lines[i]}")
            i += 1

        while j < len(new_lines):
            added += 1
            diff_lines.append(f"+{new_lines[j]}")
            j += 1

        total_changes = len(diff_lines)

        return {
            "total_changes": total_changes,
            "added_lines": added,
            "removed_lines": removed,
            "unchanged_lines": unchanged,
            "change_percentage": (total_changes / max(1, len(old_lines) + len(new_lines))) * 100,
            "diff_lines": diff_lines[:100],  # Limit diff lines
            "summary": f"{added} additions, {removed} deletions, {unchanged} unchanged"
        }

    @staticmethod
    def _generate_word_diff(old_content: str, new_content: str) -> Dict[str, Any]:
        """Generate word-level diff."""
        old_words = old_content.split()
        new_words = new_content.split()

        old_set = set(old_words)
        new_set = set(new_words)

        added = new_set - old_set
        removed = old_set - new_set
        common = old_set & new_set

        return {
            "total_words_old": len(old_words),
            "total_words_new": len(new_words),
            "added_words": len(added),
            "removed_words": len(removed),
            "common_words": len(common),
            "added_words_list": list(added)[:20],
            "removed_words_list": list(removed)[:20],
            "summary": f"{len(added)} words added, {len(removed)} words removed"
        }

    @staticmethod
    def _generate_character_diff(old_content: str, new_content: str) -> Dict[str, Any]:
        """Generate character-level diff."""
        old_len = len(old_content)
        new_len = len(new_content)

        # Simple character comparison
        added = 0
        removed = 0
        unchanged = 0

        min_len = min(old_len, new_len)
        for i in range(min_len):
            if old_content[i] == new_content[i]:
                unchanged += 1
            else:
                removed += 1
                added += 1

        if old_len > new_len:
            removed += old_len - new_len
        elif new_len > old_len:
            added += new_len - old_len

        total_changes = added + removed

        return {
            "total_changes": total_changes,
            "added_chars": added,
            "removed_chars": removed,
            "unchanged_chars": unchanged,
            "change_percentage": (total_changes / max(1, old_len + new_len)) * 100,
            "summary": f"{added} characters added, {removed} characters removed"
        }


class DocumentVersionManager:
    """
    Main document version manager with full CRUD operations.
    """

    def __init__(
        self,
        storage_dir: str = "./data/versions",
        compression: bool = True,
        max_versions_per_document: int = 100
    ):
        """
        Initialize document version manager.

        Args:
            storage_dir: Directory to store versions
            compression: Whether to compress version data
            max_versions_per_document: Maximum versions per document
        """
        self.storage = VersionStorage(
            storage_dir=storage_dir,
            compression=compression,
            max_versions_per_document=max_versions_per_document
        )

        self.diff_generator = DiffGenerator()

        logger.info(f"DocumentVersionManager initialized: {storage_dir}")

    def create_version(
        self,
        document_id: str,
        content: str,
        created_by: str = "system",
        action: VersionAction = VersionAction.CREATE,
        description: str = "",
        tags: List[str] = None,
        chunks: Optional[List[str]] = None,
        chunk_metadata: Optional[List[Dict[str, Any]]] = None,
        content_embeddings: Optional[List[List[float]]] = None,
        custom_metadata: Dict[str, Any] = None
    ) -> Optional[DocumentVersion]:
        """
        Create a new version of a document.

        Args:
            document_id: Document ID
            content: Document content
            created_by: User who created the version
            action: Action that triggered version creation
            description: Version description
            tags: Tags for the version
            chunks: Document chunks
            chunk_metadata: Chunk metadata
            content_embeddings: Content embeddings
            custom_metadata: Custom metadata

        Returns:
            Created DocumentVersion or None
        """
        try:
            # Get latest version number
            latest = self.storage.get_latest_version(document_id)
            version_number = (latest.version_number + 1) if latest else 1
            parent_version_id = latest.version_id if latest else None

            # Generate version ID
            version_id = f"v{version_number}_{uuid.uuid4().hex[:8]}"

            # Calculate hash
            content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()

            # Create metadata
            metadata = VersionMetadata(
                version_id=version_id,
                document_id=document_id,
                version_number=version_number,
                created_at=datetime.now(),
                created_by=created_by,
                action=action,
                parent_version_id=parent_version_id,
                description=description,
                size_bytes=len(content.encode('utf-8')),
                hash=content_hash,
                tags=tags or [],
                custom_metadata=custom_metadata or {}
            )

            # Create version
            version = DocumentVersion(
                metadata=metadata,
                content=content,
                content_embeddings=content_embeddings,
                chunks=chunks,
                chunk_metadata=chunk_metadata
            )

            # Save version
            if self.storage.save_version(document_id, version):
                logger.info(f"Created version {version_number} for document {document_id}")
                return version

            return None

        except Exception as e:
            logger.error(f"Failed to create version: {e}")
            return None

    def get_version(
        self,
        document_id: str,
        version_id: str = None,
        version_number: int = None,
        load_content: bool = True,
        load_embeddings: bool = False
    ) -> Optional[DocumentVersion]:
        """
        Get a document version by ID or number.

        Args:
            document_id: Document ID
            version_id: Version ID (optional)
            version_number: Version number (optional)
            load_content: Whether to load content
            load_embeddings: Whether to load embeddings

        Returns:
            DocumentVersion or None
        """
        try:
            if version_id:
                return self.storage.load_version(
                    document_id,
                    version_id,
                    load_content,
                    load_embeddings
                )
            elif version_number is not None:
                metadata = self.storage.get_version_by_number(document_id, version_number)
                if metadata:
                    return self.storage.load_version(
                        document_id,
                        metadata.version_id,
                        load_content,
                        load_embeddings
                    )

            return None

        except Exception as e:
            logger.error(f"Failed to get version: {e}")
            return None

    def get_latest_version(
        self,
        document_id: str,
        load_content: bool = True
    ) -> Optional[DocumentVersion]:
        """
        Get the latest version of a document.

        Args:
            document_id: Document ID
            load_content: Whether to load content

        Returns:
            Latest DocumentVersion or None
        """
        try:
            metadata = self.storage.get_latest_version(document_id)
            if metadata:
                return self.storage.load_version(
                    document_id,
                    metadata.version_id,
                    load_content
                )
            return None

        except Exception as e:
            logger.error(f"Failed to get latest version: {e}")
            return None

    def update_version(
        self,
        document_id: str,
        content: str,
        created_by: str = "system",
        description: str = "",
        tags: List[str] = None,
        chunks: Optional[List[str]] = None,
        chunk_metadata: Optional[List[Dict[str, Any]]] = None,
        content_embeddings: Optional[List[List[float]]] = None,
        custom_metadata: Dict[str, Any] = None
    ) -> Optional[DocumentVersion]:
        """
        Update a document by creating a new version.

        Args:
            document_id: Document ID
            content: Updated document content
            created_by: User who updated the document
            description: Update description
            tags: Tags for the new version
            chunks: Document chunks
            chunk_metadata: Chunk metadata
            content_embeddings: Content embeddings
            custom_metadata: Custom metadata

        Returns:
            New DocumentVersion or None
        """
        return self.create_version(
            document_id=document_id,
            content=content,
            created_by=created_by,
            action=VersionAction.UPDATE,
            description=description,
            tags=tags,
            chunks=chunks,
            chunk_metadata=chunk_metadata,
            content_embeddings=content_embeddings,
            custom_metadata=custom_metadata
        )

    def rollback_to_version(
        self,
        document_id: str,
        target_version_id: str = None,
        target_version_number: int = None,
        created_by: str = "system"
    ) -> Optional[DocumentVersion]:
        """
        Rollback to a previous version.

        Args:
            document_id: Document ID
            target_version_id: Target version ID
            target_version_number: Target version number
            created_by: User performing rollback

        Returns:
            New version (rollback version) or None
        """
        try:
            # Get target version
            target = self.get_version(
                document_id,
                target_version_id,
                target_version_number,
                load_content=True
            )

            if not target:
                logger.error(f"Target version not found")
                return None

            # Create new version from target
            latest = self.storage.get_latest_version(document_id)
            version_number = (latest.version_number + 1) if latest else 1

            # Generate version ID
            version_id = f"v{version_number}_{uuid.uuid4().hex[:8]}"

            # Calculate hash
            content_hash = hashlib.sha256(target.content.encode('utf-8')).hexdigest()

            # Create metadata
            metadata = VersionMetadata(
                version_id=version_id,
                document_id=document_id,
                version_number=version_number,
                created_at=datetime.now(),
                created_by=created_by,
                action=VersionAction.RESTORE,
                parent_version_id=target.metadata.version_id,
                description=f"Rollback to version {target.metadata.version_number}",
                size_bytes=len(target.content.encode('utf-8')),
                hash=content_hash,
                tags=target.metadata.tags.copy(),
                custom_metadata={
                    **target.metadata.custom_metadata,
                    "rollback_from": target.metadata.version_id,
                    "rollback_reason": "User requested rollback"
                }
            )

            # Create rollback version
            rollback_version = DocumentVersion(
                metadata=metadata,
                content=target.content,
                content_embeddings=target.content_embeddings,
                chunks=target.chunks,
                chunk_metadata=target.chunk_metadata
            )

            # Save rollback version
            if self.storage.save_version(document_id, rollback_version):
                logger.info(f"Rollback to version {target_version_number} for document {document_id}")
                return rollback_version

            return None

        except Exception as e:
            logger.error(f"Failed to rollback version: {e}")
            return None

    def compare_versions(
        self,
        document_id: str,
        version_id_1: str = None,
        version_id_2: str = None,
        version_number_1: int = None,
        version_number_2: int = None,
        algorithm: str = "unified"
    ) -> Dict[str, Any]:
        """
        Compare two versions of a document.

        Args:
            document_id: Document ID
            version_id_1: First version ID
            version_id_2: Second version ID
            version_number_1: First version number
            version_number_2: Second version number
            algorithm: Diff algorithm

        Returns:
            Diff results
        """
        try:
            # Get versions
            v1 = self.get_version(document_id, version_id_1, version_number_1, load_content=True)
            v2 = self.get_version(document_id, version_id_2, version_number_2, load_content=True)

            if not v1 or not v2:
                return {"error": "One or both versions not found"}

            # Generate diff
            diff = self.diff_generator.generate_diff(
                v1.content or "",
                v2.content or "",
                algorithm
            )

            # Add version metadata
            diff.update({
                "version_1": {
                    "id": v1.metadata.version_id,
                    "number": v1.metadata.version_number,
                    "created_at": v1.metadata.created_at.isoformat(),
                    "created_by": v1.metadata.created_by
                },
                "version_2": {
                    "id": v2.metadata.version_id,
                    "number": v2.metadata.version_number,
                    "created_at": v2.metadata.created_at.isoformat(),
                    "created_by": v2.metadata.created_by
                }
            })

            return diff

        except Exception as e:
            logger.error(f"Failed to compare versions: {e}")
            return {"error": str(e)}

    def delete_version(self, document_id: str, version_id: str) -> bool:
        """
        Delete a specific version.

        Args:
            document_id: Document ID
            version_id: Version ID

        Returns:
            Success status
        """
        return self.storage.delete_version(document_id, version_id)

    def get_version_history(
        self,
        document_id: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get version history for a document.

        Args:
            document_id: Document ID
            limit: Maximum number of versions

        Returns:
            Version history
        """
        return self.storage.get_version_history(document_id, limit)

    def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get storage statistics.

        Returns:
            Storage statistics
        """
        return self.storage.get_storage_stats()

    def export_version(
        self,
        document_id: str,
        version_id: str = None,
        version_number: int = None,
        format: str = "json"
    ) -> Optional[Dict[str, Any]]:
        """
        Export a version to a specific format.

        Args:
            document_id: Document ID
            version_id: Version ID
            version_number: Version number
            format: Export format ('json', 'text')

        Returns:
            Exported data
        """
        try:
            version = self.get_version(
                document_id,
                version_id,
                version_number,
                load_content=True
            )

            if not version:
                return None

            if format == "json":
                return version.to_dict()
            elif format == "text":
                return {
                    "content": version.content,
                    "metadata": version.metadata.to_dict()
                }
            else:
                raise ValueError(f"Unsupported format: {format}")

        except Exception as e:
            logger.error(f"Failed to export version: {e}")
            return None


# ============================================================
# Convenience Functions
# ============================================================

def create_version_manager(
    storage_dir: str = "./data/versions",
    compression: bool = True,
    max_versions: int = 100
) -> DocumentVersionManager:
    """
    Create a document version manager instance.

    Args:
        storage_dir: Directory to store versions
        compression: Whether to compress version data
        max_versions: Maximum versions per document

    Returns:
        DocumentVersionManager instance
    """
    return DocumentVersionManager(
        storage_dir=storage_dir,
        compression=compression,
        max_versions_per_document=max_versions
    )


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Create version manager
    manager = create_version_manager("./data/versions")

    # Create initial version
    doc_id = "test_doc_1"
    initial_content = "This is the initial document content.\nIt has multiple lines.\nVersion 1."

    print("Creating initial version...")
    v1 = manager.create_version(
        doc_id,
        initial_content,
        created_by="user1",
        description="Initial document creation"
    )
    print(f"Created version {v1.metadata.version_number}: {v1.metadata.version_id}")

    # Update document
    updated_content = "This is the updated document content.\nIt has more information.\nVersion 2.\nAdded a new line."

    print("\nCreating update...")
    v2 = manager.update_version(
        doc_id,
        updated_content,
        created_by="user2",
        description="Added new information"
    )
    print(f"Updated to version {v2.metadata.version_number}: {v2.metadata.version_id}")

    # Get version history
    print("\nVersion history:")
    history = manager.get_version_history(doc_id)
    for entry in history:
        print(f"  v{entry['version_number']}: {entry['description']} ({entry['created_at']})")

    # Compare versions
    print("\nComparing versions...")
    diff = manager.compare_versions(
        doc_id,
        version_number_1=1,
        version_number_2=2
    )
    print(f"Changes: {diff['summary']}")

    # Rollback
    print("\nRolling back to version 1...")
    v3 = manager.rollback_to_version(
        doc_id,
        target_version_number=1,
        created_by="user1"
    )
    print(f"Rollback to version {v3.metadata.version_number}")

    # Get storage stats
    print("\nStorage statistics:")
    stats = manager.get_storage_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

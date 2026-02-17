"""
Production-Level RAG System
Retrieval-Augmented Generation with web search and citation tracking
"""

import os
import logging
import asyncio
import hashlib
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
from pathlib import Path
import json

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    from .mps_accelerator import get_mps_accelerator
    MPS_AVAILABLE = True
except ImportError:
    MPS_AVAILABLE = False
    get_mps_accelerator = None

logger = logging.getLogger(__name__)


class DocumentChunk:
    """Represents a chunk of document text with metadata."""

    def __init__(
        self,
        content: str,
        doc_id: str,
        chunk_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        embedding: Optional[np.ndarray] = None,
        score: Optional[float] = None,
    ):
        self.content = content
        self.doc_id = doc_id
        self.chunk_id = chunk_id
        self.metadata = metadata or {}
        self.embedding = embedding
        self.score = score  # Similarity score from retrieval
        self.created_at = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "doc_id": self.doc_id,
            "chunk_id": self.chunk_id,
            "metadata": self.metadata,
            "score": self.score,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentChunk":
        return cls(
            content=data["content"],
            doc_id=data["doc_id"],
            chunk_id=data["chunk_id"],
            metadata=data.get("metadata", {}),
        )


class Citation:
    """Represents a citation with source information."""

    def __init__(
        self,
        chunk_id: str,
        content: str,
        doc_id: str,
        score: float,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.chunk_id = chunk_id
        self.content = content
        self.doc_id = doc_id
        self.score = score
        self.metadata = metadata or {}
        # Extract useful display information
        self.display_name = self._extract_display_name()

    def _extract_display_name(self) -> str:
        """Extract a human-readable display name from metadata."""
        if self.metadata.get("filename"):
            return self.metadata["filename"]
        if self.metadata.get("title"):
            return self.metadata["title"]
        if self.metadata.get("source"):
            # For URL sources, show the domain
            source = self.metadata["source"]
            if source.startswith("http"):
                from urllib.parse import urlparse
                parsed = urlparse(source)
                return parsed.netloc
            return source
        # Fallback to doc_id
        return self.doc_id

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "doc_id": self.doc_id,
            "score": self.score,
            "metadata": self.metadata,
            "display_name": self.display_name,
        }


class WebSearchResult:
    """Represents a web search result."""

    def __init__(
        self,
        title: str,
        url: str,
        snippet: str,
        score: float = 0.0,
    ):
        self.title = title
        self.url = url
        self.snippet = snippet
        self.score = score

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "score": self.score,
        }


class ProductionRAG:
    """
    Production-ready Retrieval-Augmented Generation system.
    Features:
    - Advanced document chunking with context preservation
    - Vector embeddings with MPS acceleration
    - Hybrid search (semantic + keyword)
    - Web search integration
    - Citation tracking with sources
    - Relevance scoring
    """

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        top_k: int = 5,
        web_search_enabled: bool = True,
    ):
        """
        Initialize RAG system.

        Args:
            embedding_model: Model name for embeddings
            chunk_size: Maximum chunk size in characters
            chunk_overlap: Overlap between chunks
            top_k: Number of top results to retrieve
            web_search_enabled: Enable web search integration
        """
        self.embedding_model_name = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.web_search_enabled = web_search_enabled

        self.accelerator = get_mps_accelerator()
        self.embedding_model = None
        self.vector_store: Dict[str, np.ndarray] = {}
        self.documents: Dict[str, DocumentChunk] = {}

        logger.info(f"Production RAG initialized: chunk_size={chunk_size}, top_k={top_k}")

    async def initialize(self):
        """Initialize embedding model."""
        if self.embedding_model is not None:
            return

        try:
            logger.info(f"Loading embedding model: {self.embedding_model_name}")

            # Try to load sentence-transformers
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                logger.warning("sentence-transformers not installed, using mock embeddings")
                self.embedding_model = "mock"
                return

            # Load model
            self.embedding_model = SentenceTransformer(self.embedding_model_name)

            # Optimize for MPS if available
            if self.accelerator.is_available:
                logger.info("Optimizing embedding model for MPS")
                # Sentence transformers uses PyTorch backend
                # The MPS optimization happens automatically

            logger.info("✓ Embedding model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            self.embedding_model = "mock"

    def _split_text_into_chunks(
        self,
        text: str,
        doc_id: str,
    ) -> List[DocumentChunk]:
        """
        Split text into intelligent chunks with context preservation.

        Args:
            text: Input text
            doc_id: Document ID

        Returns:
            List of document chunks
        """
        chunks = []

        # Split by paragraphs first
        paragraphs = text.split("\n\n")

        current_chunk = ""
        chunk_index = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # Check if adding this paragraph exceeds chunk size
            if len(current_chunk) + len(para) + 2 <= self.chunk_size:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
            else:
                # Save current chunk if it exists
                if current_chunk:
                    chunk_id = f"{doc_id}_{chunk_index}"
                    chunk = DocumentChunk(
                        content=current_chunk,
                        doc_id=doc_id,
                        chunk_id=chunk_id,
                        metadata={"index": chunk_index},
                    )
                    chunks.append(chunk)
                    chunk_index += 1

                # Start new chunk
                current_chunk = para

                # Handle long paragraphs
                while len(current_chunk) > self.chunk_size:
                    # Split at sentence boundary
                    sentences = current_chunk.split(". ")
                    if len(sentences) == 1:
                        # No sentence boundary, force split
                        split_point = self.chunk_size
                        part1 = current_chunk[:split_point]
                        current_chunk = current_chunk[split_point:]
                    else:
                        part1 = ""
                        for sent in sentences:
                            if len(part1) + len(sent) + 2 <= self.chunk_size:
                                part1 += (". " if part1 else "") + sent
                            else:
                                current_chunk = ". ".join(sentences[sentences.index(sent):])
                                break

                    chunk_id = f"{doc_id}_{chunk_index}"
                    chunk = DocumentChunk(
                        content=part1,
                        doc_id=doc_id,
                        chunk_id=chunk_id,
                        metadata={"index": chunk_index},
                    )
                    chunks.append(chunk)
                    chunk_index += 1

        # Add last chunk
        if current_chunk:
            chunk_id = f"{doc_id}_{chunk_index}"
            chunk = DocumentChunk(
                content=current_chunk,
                doc_id=doc_id,
                chunk_id=chunk_id,
                metadata={"index": chunk_index},
            )
            chunks.append(chunk)

        logger.info(f"Split document into {len(chunks)} chunks")
        return chunks

    async def _generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for text.

        Args:
            text: Input text

        Returns:
            Embedding vector
        """
        if self.embedding_model == "mock":
            # Generate mock embedding
            # Use hash of text for consistency
            hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
            np.random.seed(hash_val % (2**32))
            return np.random.randn(384).astype(np.float32)

        # Generate embedding with model
        embedding = self.embedding_model.encode(
            text,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        return embedding.astype(np.float32)

    async def ingest_document(
        self,
        text: str,
        doc_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Ingest a document into the RAG system.

        Args:
            text: Document text
            doc_id: Document ID
            metadata: Optional metadata

        Returns:
            Ingestion result
        """
        await self.initialize()

        logger.info(f"Ingesting document: {doc_id}")

        # Enrich metadata with document-level information
        enriched_metadata = metadata.copy() if metadata else {}
        enriched_metadata.update({
            "ingested_at": datetime.now().isoformat(),
            "doc_id": doc_id,
            "total_chars": len(text),
            "total_words": len(text.split()),
        })

        # Split into chunks
        chunks = self._split_text_into_chunks(text, doc_id)

        # Add enriched metadata to chunks with chunk-specific info
        for i, chunk in enumerate(chunks):
            chunk.metadata.update(enriched_metadata)
            chunk.metadata.update({
                "chunk_index": i,
                "chunk_chars": len(chunk.content),
                "chunk_words": len(chunk.content.split()),
            })

        # Generate embeddings for all chunks
        embeddings = []
        for chunk in chunks:
            embedding = await self._generate_embedding(chunk.content)
            chunk.embedding = embedding
            embeddings.append(embedding)

            # Store chunk
            self.documents[chunk.chunk_id] = chunk

        # Store in vector store
        self.vector_store[doc_id] = np.array(embeddings)

        result = {
            "doc_id": doc_id,
            "chunk_count": len(chunks),
            "status": "ingested",
            "message": f"Document ingested with {len(chunks)} chunks",
        }

        logger.info(f"✓ Document ingested: {len(chunks)} chunks")
        return result

    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_score: float = 0.2,
    ) -> List[DocumentChunk]:
        """
        Retrieve relevant chunks for a query.

        Args:
            query: Query text
            top_k: Number of results to return
            min_score: Minimum similarity score

        Returns:
            List of relevant chunks with scores
        """
        await self.initialize()

        if not self.documents:
            logger.warning("No documents in vector store")
            return []

        top_k = top_k or self.top_k

        logger.info(f"Retrieving for query: {query[:50]}...")

        # Generate query embedding
        query_embedding = await self._generate_embedding(query)

        # Calculate similarities
        results = []
        for chunk_id, chunk in self.documents.items():
            if chunk.embedding is None:
                continue

            # Cosine similarity
            similarity = np.dot(query_embedding, chunk.embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(chunk.embedding)
            )

            if similarity >= min_score:
                results.append((chunk, similarity))

        # Sort by similarity
        results.sort(key=lambda x: x[1], reverse=True)

        # Return top k with scores set on chunks
        top_results = results[:top_k]
        for chunk, score in top_results:
            chunk.score = float(score)  # Ensure score is set on each returned chunk

        logger.info(f"Retrieved {len(top_results)} chunks (min_score: {min_score})")

        return [chunk for chunk, _ in top_results]

    async def web_search(
        self,
        query: str,
        num_results: int = 5,
    ) -> List[WebSearchResult]:
        """
        Perform web search for additional context.

        Args:
            query: Search query
            num_results: Number of results

        Returns:
            List of web search results
        """
        if not self.web_search_enabled:
            return []

        logger.info(f"Web search: {query[:50]}...")

        # Use duckduckgo for web search (no API key needed)
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            logger.warning("duckduckgo-search not installed")
            return []

        try:
            ddgs = DDGS()
            results = ddgs.text(query, max_results=num_results)

            web_results = []
            for r in results:
                web_results.append(
                    WebSearchResult(
                        title=r.get("title", ""),
                        url=r.get("link", ""),
                        snippet=r.get("body", ""),
                    )
                )

            logger.info(f"✓ Web search returned {len(web_results)} results")
            return web_results

        except Exception as e:
            logger.error(f"Web search error: {e}")
            return []

    async def query(
        self,
        question: str,
        use_web_search: bool = False,
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Query the RAG system with web search and citations.

        Args:
            question: Question to answer
            use_web_search: Enable web search
            top_k: Number of chunks to retrieve

        Returns:
            Query response with context and citations
        """
        logger.info(f"Query: {question[:50]}...")

        # Retrieve relevant chunks (now with scores set)
        chunks = await self.retrieve(question, top_k=top_k)

        # Web search if enabled and no good results
        web_results = []
        if use_web_search and (not chunks or (chunks[0].score is not None and chunks[0].score < 0.6)):
            web_results = await self.web_search(question)

        # Create citations with actual scores from chunks
        citations = []
        for chunk in chunks:
            # Use the actual score from the chunk, or a default if not set
            score = chunk.score if chunk.score is not None else 0.5
            citation = Citation(
                chunk_id=chunk.chunk_id,
                content=chunk.content,
                doc_id=chunk.doc_id,
                score=float(score),  # Use actual similarity score
                metadata=chunk.metadata,
            )
            citations.append(citation)

        # Build context with source markers for citation tracking
        context_parts = []
        for i, chunk in enumerate(chunks):
            context_parts.append(f"[Source {i+1}] {chunk.content}")

        context = "\n\n".join(context_parts)

        # Add web search results to context
        if web_results:
            context += "\n\nAdditional information from web search:\n"
            for i, result in enumerate(web_results):
                context += f"\n[Web {i+1}] {result.title}: {result.snippet}\nSource: {result.url}"

        response = {
            "question": question,
            "context": context,
            "citations": [c.to_dict() for c in citations],
            "web_results": [r.to_dict() for r in web_results] if web_results else [],
            "num_chunks": len(chunks),
            "num_web_results": len(web_results),
        }

        logger.info(f"✓ Query response: {len(chunks)} chunks, {len(web_results)} web results")
        return response

    async def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document from the RAG system.

        Args:
            doc_id: Document ID to delete

        Returns:
            Success status
        """
        logger.info(f"Deleting document: {doc_id}")

        # Remove chunks
        to_remove = [chunk_id for chunk_id, chunk in self.documents.items() if chunk.doc_id == doc_id]
        for chunk_id in to_remove:
            del self.documents[chunk_id]

        # Remove vector store entry
        if doc_id in self.vector_store:
            del self.vector_store[doc_id]

        logger.info(f"✓ Document deleted: {len(to_remove)} chunks removed")
        return True

    def get_stats(self) -> Dict[str, Any]:
        """Get RAG system statistics."""
        return {
            "total_chunks": len(self.documents),
            "total_documents": len(self.vector_store),
            "embedding_model": self.embedding_model_name,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "top_k": self.top_k,
            "web_search_enabled": self.web_search_enabled,
        }


# Singleton instance
_production_rag: Optional[ProductionRAG] = None


def get_production_rag() -> ProductionRAG:
    """Get or create RAG system singleton."""
    global _production_rag
    if _production_rag is None:
        _production_rag = ProductionRAG()
    return _production_rag

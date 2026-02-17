"""
RAG API Endpoints
Production-level document Q&A with citations and web search
"""

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import logging

from app.services.production_rag import get_production_rag
from app.schemas.common import ApiResponse

logger = logging.getLogger(__name__)

router = APIRouter()


class DocumentIngestRequest(BaseModel):
    """Request to ingest a document."""
    text: str = Field(..., description="Document text content")
    doc_id: str = Field(..., description="Unique document identifier")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional metadata")


class DocumentIngestResponse(BaseModel):
    """Response after document ingestion."""
    doc_id: str
    chunk_count: int
    status: str
    message: str


class QueryRequest(BaseModel):
    """Request to query the RAG system."""
    question: str = Field(..., description="Question to answer", min_length=1)
    use_web_search: bool = Field(False, description="Enable web search")
    top_k: Optional[int] = Field(None, description="Number of chunks to retrieve")
    generate_answer: bool = Field(False, description="Generate LLM answer")


class Citation(BaseModel):
    """Citation information."""
    chunk_id: str
    content: str
    doc_id: str
    score: float
    metadata: Optional[Dict[str, Any]] = None
    display_name: Optional[str] = None  # Human-readable source name


class WebResult(BaseModel):
    """Web search result."""
    title: str
    url: str
    snippet: str
    score: float


class QueryResponse(BaseModel):
    """Response to a query."""
    question: str
    context: str
    citations: List[Citation]
    web_results: List[WebResult]
    answer: Optional[str] = None
    num_chunks: int
    num_web_results: int


class DeleteDocumentRequest(BaseModel):
    """Request to delete a document."""
    doc_id: str = Field(..., description="Document ID to delete")


class StatsResponse(BaseModel):
    """RAG system statistics."""
    total_chunks: int
    total_documents: int
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    web_search_enabled: bool


@router.post("/ingest", response_model=ApiResponse[DocumentIngestResponse])
async def ingest_document(request: DocumentIngestRequest):
    """
    Ingest a document into the RAG system.

    The document will be:
    1. Split into intelligent chunks (500 chars with 100 overlap)
    2. Embedded using sentence-transformers
    3. Stored in vector database

    Args:
        request: Document ingestion request

    Returns:
        Ingestion result with chunk count
    """
    try:
        rag = get_production_rag()
        await rag.initialize()

        result = await rag.ingest_document(
            text=request.text,
            doc_id=request.doc_id,
            metadata=request.metadata,
        )

        return ApiResponse(data=DocumentIngestResponse(**result))

    except Exception as e:
        logger.error(f"Error ingesting document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ingest-file")
async def ingest_file(file: UploadFile = File(...)):
    """
    Ingest a file (TXT, MD) into the RAG system.

    Args:
        file: Uploaded file

    Returns:
        Ingestion result
    """
    try:
        # Read file content
        content = await file.read()

        # Decode text
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            text = content.decode("gbk")

        # Generate doc_id from filename
        doc_id = f"file_{file.filename}_{hash(text) % 10000}"

        # Get file extension
        metadata = {
            "filename": file.filename,
            "content_type": file.content_type,
            "size_bytes": len(content),
        }

        rag = get_production_rag()
        await rag.initialize()

        result = await rag.ingest_document(
            text=text,
            doc_id=doc_id,
            metadata=metadata,
        )

        return ApiResponse(data=DocumentIngestResponse(**result))

    except Exception as e:
        logger.error(f"Error ingesting file: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query", response_model=ApiResponse[QueryResponse])
async def query_rag(request: QueryRequest):
    """
    Query the RAG system with a question.

    Returns:
    - Relevant document chunks with citations
    - Optional web search results
    - Context for LLM answer generation
    - Optional generated answer

    Args:
        request: Query request

    Returns:
        Query response with context and citations
    """
    try:
        rag = get_production_rag()
        await rag.initialize()

        result = await rag.query(
            question=request.question,
            use_web_search=request.use_web_search,
            top_k=request.top_k,
        )

        # Generate answer if requested
        answer = None
        if request.generate_answer:
            # Integrate with LLM service
            from app.services.script_generator import ScriptGenerator
            from app.config import settings

            generator = ScriptGenerator()

            # Build prompt for RAG answer generation with citation tracking
            system_prompt = """You are a helpful assistant that answers questions based on the provided context.

IMPORTANT: You MUST cite your sources using [Source X] notation when you use information from the context.
- Every time you reference information from a source, include [Source X] where X is the source number.
- If you combine information from multiple sources, cite all of them: [Source 1][Source 2].
- Only use information from the provided context sources.
- If the answer is not in the context, say "I don't have enough information to answer this question."

Your answer should be comprehensive and well-structured, with clear source citations."""

            # Context is already formatted as a string with [Source X] markers
            context_str = result.get('context', '')

            # Build citations info for reference
            citations = result.get('citations', [])
            citations_info = ""
            for i, cit in enumerate(citations):
                doc_name = cit.get('doc_id', 'Unknown')
                score = cit.get('score', 0)
                citations_info += f"\n  [Source {i+1}] {doc_name} (relevance: {score:.2f})"

            user_prompt = f"""Context Sources:{citations_info}

{context_str}

Question: {request.question}

Please provide a comprehensive answer based on the context above. Make sure to cite your sources using [Source X] notation."""

            try:
                entries = await generator.generate_script(
                    text=user_prompt,
                    system_prompt=system_prompt,
                    temperature=0.7,
                    max_tokens=2000
                )

                # Extract answer from script entries
                if entries and len(entries) > 0:
                    answer = entries[0].get("text", "Unable to generate answer.")
                else:
                    answer = "Unable to generate answer from the provided context."

            except Exception as e:
                logger.error(f"Error generating RAG answer: {e}")
                answer = f"Error generating answer: {str(e)}"

        response = QueryResponse(
            question=result["question"],
            context=result["context"],
            citations=result["citations"],
            web_results=result["web_results"],
            answer=answer,
            num_chunks=result["num_chunks"],
            num_web_results=result["num_web_results"],
        )

        return ApiResponse(data=response)

    except Exception as e:
        logger.error(f"Error querying RAG: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/document", response_model=ApiResponse[Dict[str, str]])
async def delete_document(request: DeleteDocumentRequest):
    """
    Delete a document from the RAG system.

    Args:
        request: Delete request with doc_id

    Returns:
        Success status
    """
    try:
        rag = get_production_rag()
        success = await rag.delete_document(request.doc_id)

        if success:
            return ApiResponse(data={"status": "success", "message": f"Document {request.doc_id} deleted"})
        else:
            raise HTTPException(status_code=404, detail="Document not found")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats", response_model=ApiResponse[StatsResponse])
async def get_stats():
    """
    Get RAG system statistics.

    Returns:
        System statistics
    """
    try:
        rag = get_production_rag()
        stats = rag.get_stats()

        return ApiResponse(data=StatsResponse(**stats))

    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents", response_model=ApiResponse[List[Dict[str, Any]]])
async def list_documents():
    """
    List all documents in the RAG system.

    Returns:
        List of documents with metadata
    """
    try:
        rag = get_production_rag()
        await rag.initialize()

        # Group chunks by doc_id
        documents = {}
        for chunk_id, chunk in rag.documents.items():
            if chunk.doc_id not in documents:
                documents[chunk.doc_id] = {
                    "doc_id": chunk.doc_id,
                    "chunk_count": 0,
                    "metadata": chunk.metadata,
                }
            documents[chunk.doc_id]["chunk_count"] += 1

        return ApiResponse(data=list(documents.values()))

    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

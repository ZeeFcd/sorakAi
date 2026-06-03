from __future__ import annotations

import uuid

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator


class DocumentIngestRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "filename": "main.py",
                "content": "def foo():\n    return 42\n",
                "chunk_size": 500,
                "chunk_overlap": 50,
                "mime_type": "text/x-python",
            }
        }
    )

    filename: str = Field(..., min_length=1, max_length=512)
    content: str = Field(..., min_length=1)
    chunk_size: int = Field(default=500, ge=50, le=10_000)
    chunk_overlap: int = Field(
        default=50,
        ge=0,
        le=2_000,
        description=(
            "Number of characters each chunk overlaps with its neighbour. "
            "Must be strictly less than chunk_size; the splitter enforces that."
        ),
    )
    mime_type: str | None = Field(
        default=None,
        max_length=128,
        description=(
            "Optional MIME type hint (e.g. text/x-python, text/markdown). "
            "When omitted the splitter falls back to detecting the language from "
            "the filename suffix and finally to a plain recursive character split."
        ),
    )
    document_id: str | None = Field(
        default=None,
        description="Stable id for this document; generated if omitted.",
        max_length=128,
    )
    replace_kb: bool = Field(
        default=False,
        description="If true, remove all existing knowledge before adding this document.",
    )

    @field_validator("document_id")
    @classmethod
    def doc_id_ok(cls, v: str | None) -> str | None:
        if v is not None and not v.strip():
            return None
        return v

    @field_validator("chunk_overlap")
    @classmethod
    def overlap_lt_size(cls, v: int, info: ValidationInfo) -> int:
        size = info.data.get("chunk_size")
        if isinstance(size, int) and v >= size:
            raise ValueError(f"chunk_overlap ({v}) must be < chunk_size ({size})")
        return v


class DocumentIngestResponse(BaseModel):
    message: str
    num_chunks: int
    filename: str
    document_id: str = Field(description="Id of the stored document (use for traceability)")


class DocumentSummaryResponse(BaseModel):
    """One document row in :class:`DocumentListResponse` (Wave 4)."""

    doc_id: str
    filename: str
    chunk_count: int = Field(ge=0)
    mime: str | None = None


class DocumentListResponse(BaseModel):
    documents: list[DocumentSummaryResponse]
    total: int = Field(ge=0, description="Number of distinct documents in the KB")


class DocumentDeleteResponse(BaseModel):
    """Result of ``DELETE /v1/documents/{doc_id}`` (Wave 4)."""

    doc_id: str
    removed_chunks: int = Field(ge=0)
    message: str


class QueryRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={"example": {"question": "What does foo return?", "session_id": "user-42", "top_k": 5}}
    )

    question: str = Field(..., min_length=1, max_length=4000)
    session_id: str | None = Field(
        default=None,
        description="If set, prior turns in this session are sent to the model and updated after each reply.",
        max_length=128,
    )
    top_k: int = Field(default=5, ge=1, le=20, description="Number of KB chunks to merge into context")
    use_chat_history: bool = Field(
        default=True,
        description="If false, session_id is ignored (stateless turn).",
    )


class QueryResponse(BaseModel):
    answer: str
    context_preview: str = Field(description="Short preview of merged retrieval context")
    sources_used: int = Field(default=0, description="How many KB chunks were merged into context")
    session_id: str | None = Field(default=None, description="Echo when conversation state was used")


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str


class ReadinessResponse(BaseModel):
    ready: bool
    service: str
    detail: str | None = None


def new_document_id() -> str:
    return str(uuid.uuid4())

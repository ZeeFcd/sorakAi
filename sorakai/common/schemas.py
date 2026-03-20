from pydantic import BaseModel, Field, ConfigDict


class DocumentIngestRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {"filename": "main.py", "content": "def foo():\n    return 42\n"}})

    filename: str = Field(..., min_length=1, max_length=512)
    content: str = Field(..., min_length=1)
    chunk_size: int = Field(default=500, ge=50, le=10_000)


class DocumentIngestResponse(BaseModel):
    message: str
    num_chunks: int
    filename: str


class QueryRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {"question": "What does foo return?"}})

    question: str = Field(..., min_length=1, max_length=4000)


class QueryResponse(BaseModel):
    answer: str
    context_preview: str = Field(description="Short preview of retrieved chunk")


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str


class ReadinessResponse(BaseModel):
    ready: bool
    service: str
    detail: str | None = None

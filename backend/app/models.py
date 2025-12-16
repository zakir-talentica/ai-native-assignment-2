from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime


class DocumentUploadResponse(BaseModel):
    document_id: str
    status: str
    

class QueryRequest(BaseModel):
    query: str
    conversation_id: Optional[str] = None
    

class SourceCitation(BaseModel):
    document: str
    chunk_id: str
    content: str
    score: float
    

class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceCitation]
    conversation_id: str
    

class FeedbackRequest(BaseModel):
    conversation_id: str
    turn_index: int
    feedback_type: str  # "HELPFUL" or "NOT_HELPFUL"


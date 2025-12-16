import os
import uuid
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from .models import (
    DocumentUploadResponse,
    QueryRequest,
    QueryResponse,
    FeedbackRequest,
    SourceCitation
)
from .services.document_processor import process_document
from .services.rag_engine import run_rag_query
from .services.conversation_manager import conversation_manager
from .services.external_mock import mock_expert_escalation
from .config import settings


app = FastAPI(title="RAG PoC API", version="1.0.0")

# Configure CORS for Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "message": "RAG PoC API is running"}


@app.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """Upload and process a document."""
    # Generate document ID
    document_id = f"doc_{uuid.uuid4().hex[:12]}"
    
    # Create upload directory if it doesn't exist
    os.makedirs(settings.upload_dir, exist_ok=True)
    
    # Save file
    file_path = os.path.join(settings.upload_dir, f"{document_id}_{file.filename}")
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    # Add background task to process document
    background_tasks.add_task(process_document, file_path, document_id)
    
    return DocumentUploadResponse(
        document_id=document_id,
        status="processing"
    )


@app.post("/conversations/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """Execute RAG query and manage conversation thread."""
    # Create or load conversation
    if request.conversation_id:
        conv_id = request.conversation_id
        conversation_history = conversation_manager.get_conversation_history(conv_id)
    else:
        conv_id = conversation_manager.create_conversation()
        conversation_history = []
    
    # Run RAG workflow
    result = await run_rag_query(request.query, conversation_history)
    
    # Format sources for response
    sources = [
        SourceCitation(**source)
        for source in result["sources"]
    ]
    
    # Append turn to conversation
    conversation_manager.append_turn(
        conv_id,
        request.query,
        result["answer"],
        result["sources"]
    )
    
    return QueryResponse(
        answer=result["answer"],
        sources=sources,
        conversation_id=conv_id
    )


@app.post("/feedback")
async def submit_feedback(
    request: FeedbackRequest,
    background_tasks: BackgroundTasks
):
    """Submit feedback for a query response."""
    # Store feedback in MongoDB
    conversation_manager.store_feedback(
        request.conversation_id,
        request.turn_index,
        request.feedback_type
    )
    
    # If NOT_HELPFUL, trigger escalation
    if request.feedback_type == "NOT_HELPFUL":
        # Load turn data
        conv = conversation_manager.load_conversation(request.conversation_id)
        if conv and request.turn_index < len(conv["turns"]):
            turn = conv["turns"][request.turn_index]
            background_tasks.add_task(
                mock_expert_escalation,
                turn["query"],
                turn["answer"],
                "NOT_HELPFUL"
            )
    
    return {"status": "feedback_recorded"}


@app.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Retrieve a conversation by ID."""
    conv = conversation_manager.load_conversation(conversation_id)
    if not conv:
        return {"error": "Conversation not found"}
    return conv


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


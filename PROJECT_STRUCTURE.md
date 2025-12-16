# Project Structure

This document describes the folder organization and purpose of key modules in the RAG PoC application.

## Directory Tree

```
rag-poc/
├── backend/                    # FastAPI backend application
│   ├── app/
│   │   ├── __init__.py        # Package initialization
│   │   ├── main.py            # FastAPI app and route definitions
│   │   ├── config.py          # Configuration and environment settings
│   │   ├── models.py          # Pydantic data models/schemas
│   │   └── services/          # Business logic layer
│   │       ├── __init__.py
│   │       ├── document_processor.py  # Document ingestion pipeline
│   │       ├── rag_engine.py          # LangGraph RAG workflow
│   │       ├── conversation_manager.py # MongoDB conversation CRUD
│   │       └── external_mock.py        # Mock external API integrations
│   ├── faiss_index/           # FAISS vector store (runtime created)
│   ├── uploads/               # Temporary document storage (runtime created)
│   ├── requirements.txt       # Python dependencies
│   ├── env.template           # Environment variables template
│   └── .env                   # Environment variables (not in git)
│
├── ui/                        # Streamlit frontend
│   ├── streamlit_app.py       # Main UI application
│   └── requirements.txt       # UI dependencies
│
├── evaluation/                # RAGAS evaluation framework
│   ├── ragas_eval.py          # Evaluation script
│   ├── test_queries.json      # Sample test queries
│   └── requirements.txt       # Evaluation dependencies
│
├── sample_documents/          # Sample documents for testing
│   ├── rag_overview.md
│   └── revenue_report.md
│
├── docker-compose.yml         # Docker orchestration
├── README.md                  # Main documentation
├── PROJECT_STRUCTURE.md        # This file
├── WORKFLOW_DESIGN.md         # Workflow and schema documentation
├── API-SPECIFICATION.yml       # OpenAPI specification
└── POSTMAN_COLLECTION.json    # Postman API collection
```

## Backend Structure (`backend/`)

### `app/main.py`
**Purpose**: FastAPI application entry point and route definitions

**Key Responsibilities**:
- Initialize FastAPI app with CORS middleware
- Define HTTP endpoints for:
  - Document upload (`POST /documents/upload`)
  - Query processing (`POST /conversations/query`)
  - Feedback submission (`POST /feedback`)
  - Conversation retrieval (`GET /conversations/{id}`)
- Handle request/response serialization
- Orchestrate background tasks

**Key Functions**:
- `upload_document()` - File upload handler
- `query_rag()` - RAG query execution
- `submit_feedback()` - Feedback collection
- `get_conversation()` - Conversation retrieval

### `app/config.py`
**Purpose**: Application configuration management

**Key Responsibilities**:
- Load environment variables from `.env` file
- Define settings using Pydantic BaseSettings
- Provide configuration singleton (`settings`)

**Configuration Variables**:
- `openai_api_key` - OpenAI API key (required)
- `mongodb_uri` - MongoDB connection string
- `mongodb_db` - Database name
- `faiss_index_path` - FAISS index directory path
- `upload_dir` - Document upload directory

### `app/models.py`
**Purpose**: Pydantic data models for request/response validation

**Key Models**:
- `DocumentUploadResponse` - Upload endpoint response
- `QueryRequest` - Query endpoint request
- `QueryResponse` - Query endpoint response
- `SourceCitation` - Source citation structure
- `FeedbackRequest` - Feedback endpoint request

**Benefits**:
- Type safety
- Automatic validation
- API documentation generation

### `app/services/document_processor.py`
**Purpose**: Document ingestion and processing pipeline

**Key Responsibilities**:
- Extract text from PDF, DOCX, MD files
- Chunk text using RecursiveCharacterTextSplitter
- Generate embeddings using OpenAI
- Store embeddings in FAISS index with metadata

**Key Functions**:
- `extract_text_from_pdf()` - PDF text extraction
- `extract_text_from_docx()` - DOCX text extraction
- `extract_text_from_md()` - Markdown text extraction
- `chunk_text()` - Text chunking with overlap
- `generate_embeddings_and_index()` - Embedding generation and indexing
- `load_faiss_index()` - Load existing FAISS index
- `process_document()` - Complete document processing pipeline

**Dependencies**:
- PyPDF2 for PDF processing
- python-docx for DOCX processing
- langchain-text-splitters for chunking
- langchain-openai for embeddings
- langchain-community for FAISS

### `app/services/rag_engine.py`
**Purpose**: LangGraph RAG workflow orchestration

**Key Responsibilities**:
- Define RAG state machine (TypedDict)
- Implement workflow nodes (retrieve, generate, cite)
- Build and compile LangGraph workflow
- Execute RAG queries with conversation context

**Key Components**:
- `RAGState` - State definition for workflow
- `retrieve_node()` - FAISS similarity search
- `generate_node()` - LLM answer generation
- `cite_node()` - Source citation formatting
- `build_rag_graph()` - Graph construction
- `run_rag_query()` - Workflow execution

**Workflow**:
```
Retrieve → Generate → Cite → End
```

**Dependencies**:
- langgraph for workflow orchestration
- langchain-openai for ChatOpenAI
- langchain-core for prompts

### `app/services/conversation_manager.py`
**Purpose**: MongoDB conversation thread management

**Key Responsibilities**:
- Create conversation threads
- Load conversation history
- Append query/response turns
- Store feedback
- Provide conversation context for RAG

**Key Functions**:
- `create_conversation()` - Create new thread
- `load_conversation()` - Load thread by ID
- `append_turn()` - Add query/response turn
- `get_conversation_history()` - Get history for context
- `store_feedback()` - Store user feedback

**Database Schema**:
- Collection: `conversations`
- Document structure: `{_id, created_at, turns[]}`

### `app/services/external_mock.py`
**Purpose**: Mock external API integrations

**Key Responsibilities**:
- Simulate enterprise search API
- Simulate expert escalation webhook
- Provide fallback retrieval when FAISS scores are low

**Key Functions**:
- `mock_enterprise_search()` - Simulated external search
- `mock_expert_escalation()` - Simulated escalation

**Usage**:
- Called when FAISS relevance score < 0.7
- Triggered on NOT_HELPFUL feedback

## Frontend Structure (`ui/`)

### `streamlit_app.py`
**Purpose**: Streamlit web interface

**Key Responsibilities**:
- Document upload UI
- Chat interface
- Message history display
- Source citation display
- Feedback collection UI
- Session state management

**Key Components**:
- Sidebar: Document upload
- Main area: Chat interface
- Message display: User/assistant messages
- Source expander: Citation display
- Feedback buttons: 👍/👎

**Session State**:
- `conversation_id` - Current conversation ID
- `messages` - Chat message history
- `feedback_given` - Set of feedback keys

## Evaluation Structure (`evaluation/`)

### `ragas_eval.py`
**Purpose**: RAGAS evaluation script

**Key Responsibilities**:
- Load test queries
- Prepare evaluation dataset
- Run RAGAS metrics
- Display evaluation results

**Metrics**:
- Faithfulness
- Answer Relevancy
- Context Recall
- Context Precision

### `test_queries.json`
**Purpose**: Sample test queries for evaluation

**Structure**:
```json
[
  {
    "query": "...",
    "expected_keywords": [...],
    "category": "..."
  }
]
```

## Runtime Directories

### `backend/faiss_index/`
**Purpose**: FAISS vector store persistence

**Files**:
- `index.faiss` - Vector index file
- `index.pkl` - Metadata pickle file

**Created**: Automatically on first document upload

### `backend/uploads/`
**Purpose**: Temporary document storage

**Files**: Uploaded documents before processing

**Lifecycle**: Files stored temporarily, can be cleaned periodically

## Configuration Files

### `backend/requirements.txt`
Python dependencies for backend:
- FastAPI, Uvicorn
- LangChain, LangGraph
- OpenAI SDK
- FAISS
- PyMongo
- Document processing libraries

### `ui/requirements.txt`
Python dependencies for UI:
- Streamlit
- Requests

### `evaluation/requirements.txt`
Python dependencies for evaluation:
- RAGAS
- Datasets

### `backend/env.template`
Environment variables template:
- Copy to `.env` and fill in values
- Contains all required/optional config

## Key Design Patterns

### Separation of Concerns
- **API Layer** (`main.py`): HTTP handling only
- **Service Layer** (`services/`): Business logic
- **Data Layer** (`models.py`): Data structures

### Dependency Injection
- Configuration via `settings` singleton
- Service instances (e.g., `conversation_manager`)

### Background Processing
- FastAPI BackgroundTasks for async processing
- Document processing runs in background

### State Management
- LangGraph for RAG workflow state
- MongoDB for conversation state
- Streamlit session state for UI

## Module Dependencies

```
main.py
├── config.py (settings)
├── models.py (schemas)
└── services/
    ├── document_processor.py
    │   ├── config.py (settings)
    │   └── langchain libraries
    ├── rag_engine.py
    │   ├── document_processor.py (load_faiss_index)
    │   ├── external_mock.py
    │   ├── config.py (settings)
    │   └── langgraph/langchain libraries
    ├── conversation_manager.py
    │   └── config.py (settings)
    └── external_mock.py
```

## File Naming Conventions

- **Python files**: `snake_case.py`
- **Configuration**: `.env`, `env.template`
- **Documentation**: `UPPERCASE.md` or `.md`
- **Data files**: `snake_case.json`, `snake_case.yml`

## Adding New Features

### New Document Type
1. Add extraction function to `document_processor.py`
2. Update `extract_text()` function
3. Update file validation in `main.py`

### New API Endpoint
1. Add route to `main.py`
2. Add request/response models to `models.py`
3. Implement service logic in `services/`

### New RAG Node
1. Add node function to `rag_engine.py`
2. Update `build_rag_graph()` to include node
3. Add edges to workflow

### New Database Collection
1. Add manager class to `services/`
2. Update `conversation_manager.py` or create new manager
3. Update schema documentation


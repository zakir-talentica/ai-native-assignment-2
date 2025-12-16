# RAG PoC - Retrieval-Augmented Generation Application

A lightweight proof-of-concept RAG application using FastAPI, LangChain, LangGraph, FAISS, MongoDB, and Streamlit.

## Overview

This application enables users to upload documents (PDF, DOCX, MD), ask questions, and receive AI-generated answers grounded in the uploaded content with full source citations. The system uses Retrieval-Augmented Generation (RAG) to combine semantic search with large language models for accurate, context-aware responses.

### Key Features

- **Document Ingestion**: Support for PDF, DOCX, and Markdown files
- **Semantic Search**: FAISS vector store for fast similarity search
- **RAG Pipeline**: LangGraph workflow for retrieval, generation, and citation
- **Conversation Threading**: Multi-turn conversations with context awareness
- **Source Citation**: Automatic citation of retrieved document chunks
- **Feedback Collection**: User feedback system with escalation workflow
- **Modern UI**: Streamlit-based web interface

## Architecture

```
┌─────────────┐
│  Streamlit  │  UI Layer
│     UI      │
└──────┬──────┘
       │ HTTP/REST
┌──────▼──────────────────┐
│   FastAPI Backend       │  API Layer
│  - Document Upload      │
│  - Query Processing     │
│  - Feedback Collection   │
└──────┬───────────────────┘
       │
   ┌───┴──────────────────────────┐
   │                               │
┌──▼──────────┐          ┌─────────▼────────┐
│   MongoDB   │          │   FAISS Index    │
│ (Conversations)        │   (Embeddings)    │
└─────────────┘          └──────────────────┘
```

## Prerequisites

- Python 3.9+
- MongoDB 4.4+ (or Docker)
- OpenAI API Key
- Docker & Docker Compose (optional, for containerized deployment)

## Quick Start

### Option 1: Docker Compose (Recommended)

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd rag-poc
   ```

2. **Create environment file**
   ```bash
   cp backend/env.template backend/.env
   # Edit backend/.env and add your OPENAI_API_KEY
   ```

3. **Start all services**
   ```bash
   docker-compose up -d
   ```

4. **Access the application**
   - UI: http://localhost:8501
   - API: http://localhost:8000
   - API Docs: http://localhost:8000/docs

### Option 2: Manual Setup

#### 1. MongoDB Setup

**Local Installation:**
```bash
# macOS
brew install mongodb-community
brew services start mongodb-community

# Windows
# Download from https://www.mongodb.com/try/download/community

# Linux
sudo apt-get install mongodb
sudo systemctl start mongod
```

**Docker:**
```bash
docker run -d -p 27017:27017 --name mongodb mongo
```

**Verify MongoDB:**
```bash
mongosh
# or
mongo
```

#### 2. Vector Store Setup

FAISS is automatically initialized on first document upload. No manual setup required.

**Directory Structure:**
```
backend/
├── faiss_index/     # Created automatically
│   ├── index.faiss  # Vector index
│   └── index.pkl    # Metadata
└── uploads/          # Temporary document storage
```

#### 3. Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create .env file
cp env.template .env
# Edit .env and add your OPENAI_API_KEY

# Start the backend
uvicorn app.main:app --reload
```

**Environment Variables (.env):**
```env
OPENAI_API_KEY=sk-your-key-here
MONGODB_URI=mongodb://localhost:27017
MONGODB_DB=rag_poc
FAISS_INDEX_PATH=./faiss_index
UPLOAD_DIR=./uploads
```

#### 4. Background Workers

The application uses FastAPI BackgroundTasks for document processing. No separate worker process is needed for PoC scale.

**For Production:**
- Consider Celery with Redis for distributed task processing
- See `WORKFLOW_DESIGN.md` for background task architecture

#### 5. UI Setup

```bash
# Navigate to UI directory
cd ui

# Install dependencies
pip install -r requirements.txt

# Start Streamlit
streamlit run streamlit_app.py
```

The UI will open automatically at http://localhost:8501

## Usage

### 1. Upload Documents

- Use the sidebar in the Streamlit UI
- Supported formats: PDF, DOCX, MD
- Documents are processed in the background (30-60 seconds for large files)
- Processing includes: text extraction → chunking → embedding → indexing

### 2. Ask Questions

- Type your question in the chat input
- The system will:
  1. Retrieve relevant chunks from FAISS
  2. Generate answer using GPT-3.5-turbo
  3. Return answer with source citations

### 3. Provide Feedback

- Click 👍 for helpful responses
- Click 👎 for unhelpful responses (triggers escalation)

### 4. Conversation Threading

- Conversations are automatically tracked
- Context from previous turns improves answers
- Click "Start New Conversation" to reset

## API Endpoints

See `API-SPECIFICATION.yml` for detailed API documentation.

**Key Endpoints:**
- `POST /documents/upload` - Upload document
- `POST /conversations/query` - Submit query
- `POST /feedback` - Submit feedback
- `GET /conversations/{id}` - Get conversation

Interactive API docs: http://localhost:8000/docs

## Database Schema

### MongoDB Collections

**conversations**
```json
{
  "_id": "conv_abc123",
  "created_at": "2025-01-11T10:00:00Z",
  "turns": [
    {
      "query": "What is RAG?",
      "answer": "Retrieval-Augmented Generation...",
      "sources": [...],
      "timestamp": "2025-01-11T10:00:05Z",
      "feedback": "HELPFUL"
    }
  ]
}
```

See `WORKFLOW_DESIGN.md` for complete schema documentation.

## Configuration

### Backend Configuration

Edit `backend/.env`:
- `OPENAI_API_KEY` - Required: Your OpenAI API key
- `MONGODB_URI` - Optional: MongoDB connection string (default: mongodb://localhost:27017)
- `MONGODB_DB` - Optional: Database name (default: rag_poc)
- `FAISS_INDEX_PATH` - Optional: Path to FAISS index (default: ./faiss_index)
- `UPLOAD_DIR` - Optional: Upload directory (default: ./uploads)

### UI Configuration

Edit `ui/streamlit_app.py`:
- `BACKEND_URL` - Backend API URL (default: http://localhost:8000)

## Troubleshooting

See `TROUBLESHOOTING.md` for common issues and solutions.

**Quick Fixes:**
- **MongoDB connection error**: Ensure MongoDB is running (`mongod` or Docker)
- **OpenAI API error**: Check API key in `.env` file
- **Import errors**: Install dependencies: `pip install -r requirements.txt`
- **FAISS index not found**: Normal on first run - upload a document

## Development

### Project Structure

See `PROJECT_STRUCTURE.md` for detailed folder organization.

### Running Tests

```bash
# Backend tests (when implemented)
cd backend
pytest

# Evaluation with RAGAS
cd evaluation
python ragas_eval.py
```

### Code Style

- Follow PEP 8 for Python code
- Use type hints (Pydantic models)
- Document functions with docstrings

## Performance

**Expected Performance:**
- Document Processing: 10-60 seconds (depending on size)
- Query Response: 2-6 seconds
- FAISS Retrieval: <100ms
- Concurrent Users: ~50 (PoC scale)

## Cost Estimation

**OpenAI API Costs (approximate):**
- Per document (100 pages): ~$0.005 (embeddings)
- Per query: ~$0.003 (generation)
- Monthly (100 docs, 1000 queries): ~$4

## Limitations

This is a PoC with intentional simplifications:
- No authentication
- Single FAISS index
- Basic error handling
- No rate limiting
- No distributed deployment

See `PROJECT_SUMMARY.md` for production migration path.

## License

This is a proof-of-concept application for educational purposes.

## Support

For issues or questions:
1. Check `TROUBLESHOOTING.md`
2. Review API docs at `/docs`
3. Check backend/UI logs

## Next Steps

- Upload sample documents from `sample_documents/`
- Test query functionality
- Review conversation threading
- Explore API endpoints
- Run RAGAS evaluation

---

**Built with:** FastAPI, LangChain, LangGraph, FAISS, MongoDB, OpenAI, Streamlit


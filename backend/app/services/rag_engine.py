import os
from typing import List, Optional, TypedDict
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from .document_processor import load_faiss_index
from .external_mock import mock_enterprise_search
from ..config import settings


class RAGState(TypedDict):
    query: str
    conversation_history: List[dict]
    retrieved_chunks: List[dict]
    answer: str
    sources: List[dict]


def retrieve_node(state: RAGState) -> RAGState:
    """Retrieve relevant chunks from FAISS index."""
    query = state["query"]
    
    # Load FAISS index if it exists
    if os.path.exists(f"{settings.faiss_index_path}/index.faiss"):
        vectorstore = load_faiss_index(settings.faiss_index_path)
        
        # Perform similarity search with scores
        results = vectorstore.similarity_search_with_score(query, k=5)
        
        # Format retrieved chunks
        retrieved_chunks = []
        for doc, score in results:
            retrieved_chunks.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": float(1 - score)  # Convert distance to similarity
            })
        
        # Check if scores are too low (< 0.7) and call mock enterprise search
        max_score = max([chunk["score"] for chunk in retrieved_chunks]) if retrieved_chunks else 0
        
        if max_score < 0.7:
            print(f"[RAG] Low relevance score ({max_score:.2f}), calling enterprise search...")
            external_results = mock_enterprise_search(query)
            
            # Add external results to retrieved chunks
            for ext_result in external_results:
                retrieved_chunks.append({
                    "content": ext_result["content"],
                    "metadata": {"source": ext_result["source"]},
                    "score": ext_result["score"]
                })
    else:
        # No index yet, use mock external search
        print("[RAG] No FAISS index found, using enterprise search...")
        external_results = mock_enterprise_search(query)
        retrieved_chunks = [
            {
                "content": ext_result["content"],
                "metadata": {"source": ext_result["source"]},
                "score": ext_result["score"]
            }
            for ext_result in external_results
        ]
    
    state["retrieved_chunks"] = retrieved_chunks
    return state


def generate_node(state: RAGState) -> RAGState:
    """Generate answer using LLM with retrieved context."""
    query = state["query"]
    conversation_history = state.get("conversation_history", [])
    retrieved_chunks = state["retrieved_chunks"]
    
    # Build context from retrieved chunks
    context = "\n\n".join([
        f"[Source {i+1}]: {chunk['content']}"
        for i, chunk in enumerate(retrieved_chunks[:5])
    ])
    
    # Build conversation history string
    history_str = ""
    if conversation_history:
        history_str = "Previous conversation:\n"
        for turn in conversation_history[-3:]:  # Last 3 turns for context
            history_str += f"User: {turn['query']}\nAssistant: {turn['answer']}\n\n"
    
    # Create prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant that answers questions based on the provided context.
Use the context below to answer the user's question. If the context doesn't contain enough information,
say so clearly. Always cite which sources you used in your answer."""),
        ("human", """{history}Context:
{context}

Question: {query}

Please provide a detailed answer based on the context above.""")
    ])
    
    # Generate answer
    llm = ChatOpenAI(
        model="gpt-4.1",
        temperature=0,
        openai_api_key=settings.openai_api_key
    )
    
    chain = prompt | llm
    response = chain.invoke({
        "history": history_str,
        "context": context,
        "query": query
    })
    
    state["answer"] = response.content
    return state


def cite_node(state: RAGState) -> RAGState:
    """Format source citations from retrieved chunks."""
    retrieved_chunks = state["retrieved_chunks"]
    
    sources = []
    for i, chunk in enumerate(retrieved_chunks[:5]):
        metadata = chunk["metadata"]
        sources.append({
            "document": metadata.get("filename", metadata.get("source", "Unknown")),
            "chunk_id": metadata.get("chunk_id", f"chunk_{i}"),
            "content": chunk["content"][:200] + "..." if len(chunk["content"]) > 200 else chunk["content"],
            "score": chunk["score"]
        })
    
    state["sources"] = sources
    return state


def build_rag_graph() -> StateGraph:
    """Build and compile the RAG workflow graph."""
    graph = StateGraph(RAGState)
    
    # Add nodes
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("cite", cite_node)
    
    # Define edges
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "cite")
    graph.add_edge("cite", END)
    
    return graph.compile()


# Create a single instance of the compiled graph
rag_workflow = build_rag_graph()


async def run_rag_query(query: str, conversation_history: Optional[List[dict]] = None) -> dict:
    """Execute RAG query workflow."""
    if conversation_history is None:
        conversation_history = []
    
    # Initialize state
    initial_state = {
        "query": query,
        "conversation_history": conversation_history,
        "retrieved_chunks": [],
        "answer": "",
        "sources": []
    }
    
    # Execute workflow
    result = rag_workflow.invoke(initial_state)
    
    return {
        "answer": result["answer"],
        "sources": result["sources"]
    }


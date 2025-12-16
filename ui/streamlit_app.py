import streamlit as st
import requests
from typing import Optional


BACKEND_URL = "http://localhost:8000"


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="RAG PoC - Document Q&A",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("📚 RAG PoC - Document Q&A")
    st.markdown("Upload documents and ask questions using Retrieval-Augmented Generation")
    
    # Initialize session state
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "feedback_given" not in st.session_state:
        st.session_state.feedback_given = set()
    
    # Sidebar: Document Upload
    with st.sidebar:
        st.header("📤 Upload Documents")
        st.markdown("Supported formats: PDF, DOCX, MD")
        
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=["pdf", "docx", "md"],
            help="Upload documents to add them to the knowledge base"
        )
        
        if uploaded_file is not None:
            if st.button("Upload", type="primary"):
                with st.spinner("Uploading and processing..."):
                    try:
                        files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                        response = requests.post(
                            f"{BACKEND_URL}/documents/upload",
                            files=files,
                            timeout=30
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            st.success(f"✅ Uploaded: {result['document_id']}")
                            st.info("Document is being processed in the background.")
                        else:
                            st.error(f"Upload failed: {response.text}")
                    except Exception as e:
                        st.error(f"Error uploading file: {str(e)}")
        
        st.markdown("---")
        
        # Conversation controls
        st.header("💬 Conversation")
        if st.button("Start New Conversation"):
            st.session_state.conversation_id = None
            st.session_state.messages = []
            st.session_state.feedback_given = set()
            st.rerun()
        
        if st.session_state.conversation_id:
            st.text(f"ID: {st.session_state.conversation_id[:8]}...")
        else:
            st.text("No active conversation")
    
    # Main: Chat Interface
    st.markdown("### 💭 Chat")
    
    # Display chat history
    for idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
            if msg["role"] == "assistant":
                # Show sources in expander
                sources = msg.get("sources", [])
                if sources:
                    with st.expander(f"📎 Sources ({len(sources)})"):
                        for i, src in enumerate(sources):
                            st.markdown(f"**Source {i+1}: {src['document']}** (relevance: {src['score']:.2f})")
                            st.text(src["content"])
                            st.markdown("---")
                
                # Feedback buttons
                feedback_key = f"feedback_{idx}"
                if feedback_key not in st.session_state.feedback_given:
                    col1, col2, col3 = st.columns([1, 1, 8])
                    
                    with col1:
                        if st.button("👍", key=f"helpful_{idx}", help="Mark as helpful"):
                            submit_feedback(idx, "HELPFUL")
                            st.session_state.feedback_given.add(feedback_key)
                            st.rerun()
                    
                    with col2:
                        if st.button("👎", key=f"not_helpful_{idx}", help="Mark as not helpful"):
                            submit_feedback(idx, "NOT_HELPFUL")
                            st.session_state.feedback_given.add(feedback_key)
                            st.rerun()
                else:
                    st.caption("✓ Feedback recorded")
    
    # Query input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = requests.post(
                        f"{BACKEND_URL}/conversations/query",
                        json={
                            "query": prompt,
                            "conversation_id": st.session_state.conversation_id
                        },
                        timeout=60
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Update conversation_id
                        st.session_state.conversation_id = result["conversation_id"]
                        
                        # Display answer
                        st.markdown(result["answer"])
                        
                        # Add assistant message to chat
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": result["answer"],
                            "sources": result["sources"]
                        })
                        
                        st.rerun()
                    else:
                        st.error(f"Query failed: {response.text}")
                
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.info("Make sure the backend server is running on http://localhost:8000")


def submit_feedback(message_idx: int, feedback_type: str):
    """Submit feedback for a specific message."""
    try:
        # Calculate turn index (each turn has user + assistant message)
        turn_index = message_idx // 2
        
        response = requests.post(
            f"{BACKEND_URL}/feedback",
            json={
                "conversation_id": st.session_state.conversation_id,
                "turn_index": turn_index,
                "feedback_type": feedback_type
            },
            timeout=10
        )
        
        if response.status_code == 200:
            if feedback_type == "HELPFUL":
                st.toast("✅ Thank you for your feedback!", icon="👍")
            else:
                st.toast("📢 Feedback recorded. Escalating to expert.", icon="👎")
        else:
            st.error(f"Failed to submit feedback: {response.text}")
    
    except Exception as e:
        st.error(f"Error submitting feedback: {str(e)}")


if __name__ == "__main__":
    main()


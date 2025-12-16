from pymongo import MongoClient
from datetime import datetime
from typing import List, Optional
import uuid
from ..config import settings


class ConversationManager:
    """Manages conversation threads in MongoDB."""
    
    def __init__(self):
        self.client = MongoClient(settings.mongodb_uri)
        self.db = self.client[settings.mongodb_db]
        self.conversations = self.db.conversations
    
    def create_conversation(self) -> str:
        """Create a new conversation thread."""
        conv_id = f"conv_{uuid.uuid4().hex[:12]}"
        self.conversations.insert_one({
            "_id": conv_id,
            "created_at": datetime.utcnow(),
            "turns": []
        })
        return conv_id
    
    def load_conversation(self, conv_id: str) -> Optional[dict]:
        """Load conversation from MongoDB."""
        return self.conversations.find_one({"_id": conv_id})
    
    def append_turn(self, conv_id: str, query: str, answer: str, sources: List[dict]) -> None:
        """Append a new turn to the conversation."""
        turn = {
            "query": query,
            "answer": answer,
            "sources": sources,
            "timestamp": datetime.utcnow()
        }
        self.conversations.update_one(
            {"_id": conv_id},
            {"$push": {"turns": turn}}
        )
    
    def get_conversation_history(self, conv_id: str) -> List[dict]:
        """Get conversation history for context."""
        conv = self.load_conversation(conv_id)
        if not conv:
            return []
        
        return [
            {"query": turn["query"], "answer": turn["answer"]}
            for turn in conv.get("turns", [])
        ]
    
    def store_feedback(self, conv_id: str, turn_index: int, feedback_type: str) -> None:
        """Store feedback for a specific turn."""
        self.conversations.update_one(
            {"_id": conv_id},
            {"$set": {f"turns.{turn_index}.feedback": feedback_type}}
        )


# Create singleton instance
conversation_manager = ConversationManager()


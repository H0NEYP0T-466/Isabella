"""Service layer for chat operations."""
import logging
from typing import List, Dict
from datetime import datetime
from config.database import Database

logger = logging.getLogger(__name__)

class ChatService:
    """Service for managing chat messages in MongoDB."""
    
    @staticmethod
    async def save_message(role: str, content: str, thinking: bool = False, model: str = None) -> Dict:
        """Save a chat message to the database."""
        try:
            db = Database.get_db()
            message = {
                "role": role,
                "content": content,
                "timestamp": datetime.utcnow(),
                "thinking": thinking,
                "model": model
            }
            result = await db.chats.insert_one(message)
            message["_id"] = str(result.inserted_id)
            logger.info(f"✓ Message saved to database - Role: {role}, Length: {len(content)} chars")
            return message
        except Exception as e:
            logger.error(f"✗ Error saving message to database: {str(e)}")
            raise

    @staticmethod
    async def get_recent_messages(limit: int = 50) -> List[Dict]:
        """Fetch recent messages from the database."""
        try:
            db = Database.get_db()
            cursor = db.chats.find().sort("timestamp", -1).limit(limit)
            messages = await cursor.to_list(length=limit)
            
            # Reverse to get chronological order
            messages.reverse()
            
            # Convert ObjectId to string for JSON serialization
            for msg in messages:
                msg["_id"] = str(msg["_id"])
                # Convert datetime to ISO format string
                if isinstance(msg.get("timestamp"), datetime):
                    msg["timestamp"] = msg["timestamp"].isoformat()
            
            logger.info(f"✓ Retrieved {len(messages)} messages from database")
            return messages
        except Exception as e:
            logger.error(f"✗ Error fetching messages from database: {str(e)}")
            raise

    @staticmethod
    async def get_context_messages(limit: int = 10) -> List[Dict]:
        """Get the last N messages for context window."""
        try:
            db = Database.get_db()
            cursor = db.chats.find().sort("timestamp", -1).limit(limit)
            messages = await cursor.to_list(length=limit)
            
            # Reverse to get chronological order
            messages.reverse()
            
            # Format for API context
            context = [
                {"role": msg["role"], "content": msg["content"]}
                for msg in messages
            ]
            
            logger.info(f"✓ Retrieved {len(context)} messages for context window")
            return context
        except Exception as e:
            logger.error(f"✗ Error fetching context messages: {str(e)}")
            return []

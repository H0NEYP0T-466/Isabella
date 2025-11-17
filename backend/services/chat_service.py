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
    
    @staticmethod
    async def get_last_user_message_timestamp() -> datetime:
        """Get the timestamp of the last user message."""
        try:
            db = Database.get_db()
            cursor = db.chats.find({"role": "user"}).sort("timestamp", -1).limit(1)
            messages = await cursor.to_list(length=1)
            
            if messages:
                return messages[0]["timestamp"]
            return None
        except Exception as e:
            logger.error(f"✗ Error fetching last user message timestamp: {str(e)}")
            return None
    
    @staticmethod
    def calculate_time_gap_category(time_gap_seconds: float) -> str:
        """
        Calculate the category of time gap for emotional response.
        
        Returns:
            - "short": less than 1 hour
            - "long": 1-24 hours  
            - "very_long": 1-3 days
            - "extremely_long": more than 3 days
        """
        hours = time_gap_seconds / 3600
        days = time_gap_seconds / 86400
        
        if hours < 1:
            return "short"
        elif hours < 24:
            return "long"
        elif days < 3:
            return "very_long"
        else:
            return "extremely_long"
    
    @staticmethod
    def format_time_gap_for_human(time_gap_seconds: float) -> str:
        """Format time gap in human-readable form."""
        hours = time_gap_seconds / 3600
        days = time_gap_seconds / 86400
        
        if hours < 1:
            minutes = int(time_gap_seconds / 60)
            return f"{minutes} minutes"
        elif hours < 24:
            return f"{int(hours)} hours"
        else:
            return f"{int(days)} days"

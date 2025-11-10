"""Database configuration for MongoDB connection."""
import logging
from motor.motor_asyncio import AsyncIOMotorClient
from typing import Optional

logger = logging.getLogger(__name__)

class Database:
    client: Optional[AsyncIOMotorClient] = None
    db = None

    @classmethod
    async def connect_db(cls):
        """Establish connection to MongoDB."""
        try:
            mongodb_url = "mongodb://127.0.0.1:27017/isabella"
            cls.client = AsyncIOMotorClient(mongodb_url)
            cls.db = cls.client.isabella
            
            # Test the connection
            await cls.client.admin.command('ping')
            logger.info(f"✓ MongoDB connection established: {mongodb_url}")
            logger.info(f"✓ Connected to database: isabella")
        except Exception as e:
            logger.error(f"✗ Failed to connect to MongoDB: {str(e)}")
            raise

    @classmethod
    async def close_db(cls):
        """Close database connection."""
        if cls.client:
            cls.client.close()
            logger.info("✓ MongoDB connection closed")

    @classmethod
    def get_db(cls):
        """Get database instance."""
        if cls.db is None:
            raise Exception("Database not initialized. Call connect_db() first.")
        return cls.db

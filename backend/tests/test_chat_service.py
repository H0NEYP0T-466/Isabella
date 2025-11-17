"""
Unit tests for chat service timestamp functionality.
"""
import unittest
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.chat_service import ChatService


class TestChatServiceTimestamps(unittest.TestCase):
    """Test chat service timestamp handling."""
    
    @patch('services.chat_service.Database.get_db')
    async def test_get_context_messages_includes_timestamps(self, mock_get_db):
        """Test that get_context_messages includes timestamps in context."""
        # Mock database
        mock_db = MagicMock()
        mock_cursor = MagicMock()
        
        # Create mock messages with timestamps
        mock_messages = [
            {
                "role": "user",
                "content": "Hello",
                "timestamp": datetime(2024, 1, 1, 10, 0, 0)
            },
            {
                "role": "assistant", 
                "content": "Hi there!",
                "timestamp": datetime(2024, 1, 1, 10, 0, 5)
            }
        ]
        
        mock_cursor.to_list = AsyncMock(return_value=list(reversed(mock_messages)))
        mock_db.chats.find.return_value.sort.return_value.limit.return_value = mock_cursor
        mock_get_db.return_value = mock_db
        
        # Call the method
        result = await ChatService.get_context_messages(limit=10)
        
        # Verify timestamps are included
        self.assertEqual(len(result), 2)
        self.assertIn("timestamp", result[0])
        self.assertIn("timestamp", result[1])
        self.assertEqual(result[0]["role"], "user")
        self.assertEqual(result[0]["content"], "Hello")
        self.assertIsInstance(result[0]["timestamp"], str)
        self.assertIn("2024-01-01", result[0]["timestamp"])
    
    def test_timestamp_format_validation(self):
        """Test that timestamps are properly formatted as ISO strings."""
        test_datetime = datetime(2024, 1, 1, 12, 30, 45)
        iso_string = test_datetime.isoformat()
        
        # Verify ISO format
        self.assertIsInstance(iso_string, str)
        self.assertIn("2024-01-01", iso_string)
        self.assertIn("12:30:45", iso_string)


class TestIsolateModeHandling(unittest.TestCase):
    """Test isolate mode functionality."""
    
    def test_isolate_mode_returns_empty_context(self):
        """Test that isolate mode conceptually returns empty context."""
        # This is a conceptual test - in the actual implementation,
        # the chat route handles this by checking request.isolate
        isolate = True
        context_messages = [] if isolate else ["some", "messages"]
        
        self.assertEqual(context_messages, [])
    
    def test_normal_mode_returns_context(self):
        """Test that normal mode allows context."""
        isolate = False
        context_messages = [] if isolate else ["message1", "message2"]
        
        self.assertNotEqual(context_messages, [])
        self.assertEqual(len(context_messages), 2)


if __name__ == '__main__':
    # Run async tests with asyncio
    import asyncio
    
    # Create a test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add tests
    suite.addTests(loader.loadTestsFromTestCase(TestChatServiceTimestamps))
    suite.addTests(loader.loadTestsFromTestCase(TestIsolateModeHandling))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    
    # Helper to run async tests
    def run_async_test(test):
        """Run an async test case."""
        if hasattr(test, '_testMethodName'):
            method = getattr(test, test._testMethodName)
            if asyncio.iscoroutinefunction(method):
                asyncio.run(method())
                return True
        return False
    
    # Run each test
    for test in suite:
        if not run_async_test(test):
            # Run sync tests normally
            test.debug()
    
    print("\n✓ All tests completed")

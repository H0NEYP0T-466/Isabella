"""
Unit tests for emotion detection integration.
"""
import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_models.emotion_detector_model.preTrainedModel.robertA_model import (
    format_emotions,
    predict_emotions
)


class TestEmotionFormatting(unittest.TestCase):
    """Test emotion formatting utilities."""
    
    def test_format_emotions_single(self):
        """Test formatting a single emotion."""
        emotions = [("joy", 0.72)]
        result = format_emotions(emotions)
        self.assertIn("joy", result)
        self.assertIn("0.72", result)
    
    def test_format_emotions_multiple(self):
        """Test formatting multiple emotions."""
        emotions = [("joy", 0.72), ("excitement", 0.65), ("gratitude", 0.58)]
        result = format_emotions(emotions)
        self.assertIn("joy", result)
        self.assertIn("excitement", result)
        self.assertIn("gratitude", result)
        self.assertIn("0.72", result)
        self.assertIn("0.65", result)
        self.assertIn("0.58", result)
    
    def test_format_emotions_empty(self):
        """Test formatting empty emotion list."""
        emotions = []
        result = format_emotions(emotions)
        self.assertIn("neutral", result)
    
    def test_format_emotions_confidence_format(self):
        """Test that confidence scores are formatted to 2 decimal places."""
        emotions = [("joy", 0.7234567)]
        result = format_emotions(emotions)
        self.assertIn("0.72", result)
        self.assertNotIn("0.7234567", result)


class TestEmotionPrediction(unittest.TestCase):
    """Test emotion prediction functionality."""
    
    def test_predict_emotions_empty_text(self):
        """Test that empty text returns empty list."""
        result = predict_emotions("")
        self.assertEqual(result, [])
    
    def test_predict_emotions_whitespace(self):
        """Test that whitespace-only text returns empty list."""
        result = predict_emotions("   ")
        self.assertEqual(result, [])
    
    @patch('ml_models.emotion_detector_model.preTrainedModel.robertA_model.get_model')
    def test_predict_emotions_returns_list(self, mock_get_model):
        """Test that predict_emotions returns a list of tuples."""
        # Mock the model
        mock_model = MagicMock()
        mock_model.predict.return_value = [("joy", 0.75), ("excitement", 0.62)]
        mock_get_model.return_value = mock_model
        
        result = predict_emotions("I'm so happy today!")
        
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) > 0)
        # Check tuple structure
        for item in result:
            self.assertIsInstance(item, tuple)
            self.assertEqual(len(item), 2)
            self.assertIsInstance(item[0], str)  # label
            self.assertIsInstance(item[1], float)  # score
    
    @patch('ml_models.emotion_detector_model.preTrainedModel.robertA_model.get_model')
    def test_predict_emotions_handles_exception(self, mock_get_model):
        """Test that exceptions are handled gracefully."""
        # Mock the model to raise an exception
        mock_model = MagicMock()
        mock_model.predict.side_effect = Exception("Model error")
        mock_get_model.return_value = mock_model
        
        # Should return empty list instead of raising exception
        result = predict_emotions("test text")
        self.assertEqual(result, [])
    
    @patch('ml_models.emotion_detector_model.preTrainedModel.robertA_model.get_model')
    def test_predict_emotions_with_threshold(self, mock_get_model):
        """Test that threshold parameter is passed correctly."""
        mock_model = MagicMock()
        mock_model.predict.return_value = [("joy", 0.85)]
        mock_get_model.return_value = mock_model
        
        predict_emotions("test", threshold=0.5)
        
        # Verify threshold was passed to model
        mock_model.predict.assert_called_once()
        call_args = mock_model.predict.call_args
        self.assertEqual(call_args.kwargs['threshold'], 0.5)
    
    @patch('ml_models.emotion_detector_model.preTrainedModel.robertA_model.get_model')
    def test_predict_emotions_with_top_k(self, mock_get_model):
        """Test that top_k parameter is passed correctly."""
        mock_model = MagicMock()
        mock_model.predict.return_value = [("joy", 0.85), ("excitement", 0.75)]
        mock_get_model.return_value = mock_model
        
        predict_emotions("test", top_k=3)
        
        # Verify top_k was passed to model
        mock_model.predict.assert_called_once()
        call_args = mock_model.predict.call_args
        self.assertEqual(call_args.kwargs['top_k'], 3)


class TestSystemPromptAugmentation(unittest.TestCase):
    """Test system prompt augmentation with emotions."""
    
    def test_system_prompt_contains_emotions(self):
        """Test that formatted emotions appear in system prompt context."""
        emotions = [("joy", 0.72), ("excitement", 0.65)]
        formatted = format_emotions(emotions)
        
        # Build the emotion context as it would appear in chat.py
        emotion_context = f"""

Current user emotional state (multi-label):
{formatted}
Guidelines:
1. Acknowledge the expressed emotions succinctly and naturally.
2. Adjust tone: supportive for distress, encouraging for anxiety/anticipation, celebratory for positive emotions.
3. Do not infer emotions not listed.
4. Preserve all existing system rules and persona instructions."""
        
        # Verify key components are present
        self.assertIn("Current user emotional state", emotion_context)
        self.assertIn("joy", emotion_context)
        self.assertIn("excitement", emotion_context)
        self.assertIn("0.72", emotion_context)
        self.assertIn("0.65", emotion_context)
        self.assertIn("Guidelines:", emotion_context)


if __name__ == '__main__':
    unittest.main()

"""
RoBERTa-based emotion detection model for multi-label emotion classification.
Uses SamLowe/roberta-base-go_emotions pretrained model.
"""
import logging
import os
import time
from typing import Optional
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)

MODEL_NAME = "SamLowe/roberta-base-go_emotions"

EMOTION_LABELS_28 = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
    'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]


class EmotionDetectorModel:
    """Singleton class for emotion detection using RoBERTa model."""
    
    _instance: Optional['EmotionDetectorModel'] = None
    _initialized: bool = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the emotion detection model (lazy loading)."""
        if not EmotionDetectorModel._initialized:
            try:
                logger.info("🔄 Loading emotion detection model...")
                start_time = time.time()
                
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                logger.info(f"Using device: {self.device}")
                
                self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
                self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
                self.model.to(self.device)
                self.model.eval()
                
                load_time = time.time() - start_time
                logger.info(f"✅ Emotion detection model loaded successfully in {load_time:.2f}s")
                
                EmotionDetectorModel._initialized = True
                
            except Exception as e:
                logger.error(f"❌ Failed to load emotion detection model: {str(e)}")
                raise
    
    def predict(self, text: str, threshold: float = 0.35, top_k: Optional[int] = None) -> list[tuple[str, float]]:
        """
        Predict emotions from text using multi-label classification.
        
        Args:
            text: Input text to analyze
            threshold: Minimum confidence threshold for including emotions (default: 0.35)
            top_k: Optional limit on number of emotions to return (None = no limit)
            
        Returns:
            List of (emotion_label, confidence_score) tuples, sorted by confidence descending
        """
        if not text or not text.strip():
            return []
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                text,
                max_length=512,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use sigmoid for multi-label classification
                probs = torch.sigmoid(outputs.logits).cpu().numpy()[0]
            
            # Filter emotions above threshold
            emotions = [
                (EMOTION_LABELS_28[i], float(probs[i]))
                for i in range(len(EMOTION_LABELS_28))
                if probs[i] >= threshold
            ]
            
            # Sort by confidence descending
            emotions.sort(key=lambda x: x[1], reverse=True)
            
            # If no emotions above threshold, include top-1
            if not emotions:
                max_idx = probs.argmax()
                emotions = [(EMOTION_LABELS_28[max_idx], float(probs[max_idx]))]
            
            # Apply top_k limit if specified
            if top_k is not None and top_k > 0:
                emotions = emotions[:top_k]
            
            return emotions
            
        except Exception as e:
            logger.error(f"❌ Error during emotion prediction: {str(e)}")
            raise


# Global singleton instance (lazy loaded)
_model_instance: Optional[EmotionDetectorModel] = None


def get_model() -> EmotionDetectorModel:
    """Get or create the singleton emotion detection model instance."""
    global _model_instance
    if _model_instance is None:
        _model_instance = EmotionDetectorModel()
    return _model_instance


def predict_emotions(text: str, threshold: float = 0.35, top_k: Optional[int] = None) -> list[tuple[str, float]]:
    """
    Module-level function to predict emotions from text.
    
    Args:
        text: Input text to analyze
        threshold: Minimum confidence threshold for including emotions (default: 0.35)
        top_k: Optional limit on number of emotions to return (None = no limit)
        
    Returns:
        List of (emotion_label, confidence_score) tuples, sorted by confidence descending
        Returns empty list if prediction fails or text is empty.
    """
    if not text or not text.strip():
        return []
    
    try:
        model = get_model()
        return model.predict(text, threshold=threshold, top_k=top_k)
    except Exception as e:
        logger.error(f"❌ Failed to predict emotions: {str(e)}")
        return []


def format_emotions(emotions: list[tuple[str, float]]) -> str:
    """
    Format emotion list for system prompt injection.
    
    Args:
        emotions: List of (label, confidence) tuples
        
    Returns:
        Formatted string with one emotion per line
    """
    if not emotions:
        return "- neutral (no strong emotions detected)"
    
    lines = []
    for label, score in emotions:
        lines.append(f"- {label} ({score:.2f})")
    
    return "\n".join(lines)

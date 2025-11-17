"""
Demo script to show emotion detection functionality (without requiring torch installation).
This shows the expected behavior and integration flow.
"""

# Mock the emotion detection result to demonstrate the flow
def mock_predict_emotions(text: str, threshold: float = 0.35, top_k=None):
    """Mock emotion prediction for demonstration."""
    # Simulate different emotion patterns based on text content
    text_lower = text.lower()
    emotions = []
    
    if any(word in text_lower for word in ['happy', 'joy', 'excited', 'great', 'awesome']):
        emotions = [('joy', 0.85), ('excitement', 0.72), ('optimism', 0.65)]
    elif any(word in text_lower for word in ['sad', 'unhappy', 'depressed', 'down']):
        emotions = [('sadness', 0.82), ('disappointment', 0.68)]
    elif any(word in text_lower for word in ['angry', 'mad', 'furious', 'annoyed']):
        emotions = [('anger', 0.88), ('annoyance', 0.75)]
    elif any(word in text_lower for word in ['worried', 'anxious', 'nervous', 'scared']):
        emotions = [('fear', 0.76), ('nervousness', 0.69), ('anxiety', 0.62)]
    elif any(word in text_lower for word in ['confused', 'unsure', 'puzzled']):
        emotions = [('confusion', 0.79), ('curiosity', 0.58)]
    elif any(word in text_lower for word in ['thank', 'grateful', 'appreciate']):
        emotions = [('gratitude', 0.91), ('admiration', 0.64)]
    else:
        emotions = [('neutral', 0.45)]
    
    # Apply threshold filtering
    emotions = [(label, score) for label, score in emotions if score >= threshold]
    
    # Apply top_k if specified
    if top_k:
        emotions = emotions[:top_k]
    
    return emotions


def format_emotions(emotions):
    """Format emotion list for system prompt injection."""
    if not emotions:
        return "- neutral (no strong emotions detected)"
    
    lines = []
    for label, score in emotions:
        lines.append(f"- {label} ({score:.2f})")
    
    return "\n".join(lines)


def demo_emotion_integration():
    """Demonstrate the emotion detection and system prompt augmentation."""
    
    test_messages = [
        "I'm so happy and excited about this new project!",
        "I'm feeling really sad and down today.",
        "This is so frustrating! I can't get it to work!",
        "I'm worried about the presentation tomorrow.",
        "Thank you so much for your help! I really appreciate it.",
        "What is the weather like today?",
    ]
    
    print("=" * 80)
    print("EMOTION DETECTION DEMO")
    print("=" * 80)
    print()
    
    for i, message in enumerate(test_messages, 1):
        print(f"\n{'─' * 80}")
        print(f"Example {i}: {message}")
        print('─' * 80)
        
        # Simulate emotion detection
        emotions = mock_predict_emotions(message)
        
        if emotions:
            print(f"\n🎭 Detected Emotions:")
            for emotion, confidence in emotions:
                print(f"   - {emotion}: {confidence:.2f}")
            
            # Show how emotions would be formatted in system prompt
            formatted = format_emotions(emotions)
            
            print(f"\n📝 System Prompt Addition:")
            emotion_context = f"""
Current user emotional state (multi-label):
{formatted}
Guidelines:
1. Acknowledge the expressed emotions succinctly and naturally.
2. Adjust tone: supportive for distress, encouraging for anxiety/anticipation, celebratory for positive emotions.
3. Do not infer emotions not listed.
4. Preserve all existing system rules and persona instructions."""
            
            print(emotion_context)
        else:
            print("\nℹ️ No strong emotions detected (would skip augmentation)")
    
    print("\n" + "=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)
    
    # Show environment variable configuration
    print("\n📋 Environment Configuration:")
    print("  EMOTION_ENABLE=true      # Enable/disable emotion detection")
    print("  EMOTION_THRESHOLD=0.35   # Minimum confidence threshold")
    print("  EMOTION_TOP_K=           # Optional: limit number of emotions (empty = no limit)")


if __name__ == "__main__":
    demo_emotion_integration()

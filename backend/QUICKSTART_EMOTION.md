# Emotion Detection - Quick Start Guide

## 🚀 Installation (30 seconds)

```bash
# 1. Install dependencies
cd backend
pip install -r requirements.txt

# 2. Configure (optional - already enabled by default)
cp .env.example .env
# Edit .env if you want to customize:
#   EMOTION_ENABLE=true
#   EMOTION_THRESHOLD=0.35
#   EMOTION_TOP_K= (leave empty for no limit)

# 3. Start the server
uvicorn main:app --host 0.0.0.0 --port 5000 --reload
```

## ✅ What's Working

When a user sends a message, the system now:

1. ✅ Detects emotions automatically (28 categories)
2. ✅ Logs detected emotions with confidence scores
3. ✅ Injects emotions into system prompt for LLM
4. ✅ Times each stage (emotion, LLM, TTS)
5. ✅ Logs performance summary
6. ✅ Handles errors gracefully (never blocks responses)

## 📊 Example Log Output

```
================================================================================
📨 NEW CHAT REQUEST
User message: I'm so excited about this project!
Thinking mode: false
Selected model: LongCat-Flash-Chat
🎭 Running emotion detection...
✅ Detected emotions (took 0.125s):
   - joy: 0.85
   - excitement: 0.78
   - optimism: 0.62
🔄 Calling LongCat API...
✓ Received AI response - Length: 156 chars (took 1.234s)
AI response preview: That's wonderful! Your excitement is contagious! 🎉...
🔊 Generating TTS audio for AI response...
✓ TTS audio generated: speech_abc123.wav (took 2.456s)
✓ Chat request completed successfully
⏱️ Performance Summary: Emotion=0.125s | LLM=1.234s | TTS=2.456s | Total=3.815s
================================================================================
```

## 🎭 Emotion Categories

28 emotions detected:
- **Positive**: joy, excitement, gratitude, admiration, optimism, love, pride...
- **Negative**: sadness, anger, fear, disappointment, disgust, grief...
- **Neutral**: confusion, curiosity, surprise, realization, neutral

## ⚙️ Configuration

### Enable/Disable
```bash
# Enable (default)
EMOTION_ENABLE=true

# Disable
EMOTION_ENABLE=false
```

### Adjust Sensitivity
```bash
# More sensitive (more emotions detected)
EMOTION_THRESHOLD=0.25

# Default
EMOTION_THRESHOLD=0.35

# Less sensitive (fewer, stronger emotions only)
EMOTION_THRESHOLD=0.50
```

### Limit Results
```bash
# Top 3 emotions only
EMOTION_TOP_K=3

# No limit (default)
# EMOTION_TOP_K= (leave empty)
```

## 🧪 Test It

### Run Demo (no model required)
```bash
python demo_emotion_detection.py
```

### Run Unit Tests
```bash
python -m unittest tests.test_emotion_integration -v
```

### Manual Test
Send a chat request via the API:
```bash
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "I am so happy today!", "thinking": false}'
```

Check the server logs to see detected emotions!

## 📝 System Prompt Augmentation

When emotions are detected, the system adds this to the LLM prompt:

```
Current user emotional state (multi-label):
- joy (0.85)
- excitement (0.78)
Guidelines:
1. Acknowledge the expressed emotions succinctly and naturally.
2. Adjust tone: supportive for distress, encouraging for anxiety/anticipation, celebratory for positive emotions.
3. Do not infer emotions not listed.
4. Preserve all existing system rules and persona instructions.
```

## ⚡ Performance

- **First run**: ~5 seconds (downloads model from Hugging Face)
- **Subsequent runs**: <1 second startup
- **Per message**: 50-500ms depending on GPU/CPU

### Hardware
- **With GPU**: ~50-150ms per message
- **Without GPU**: ~200-500ms per message

## 🔧 Troubleshooting

### Model won't download
- Check internet connection
- Hugging Face should be accessible
- First download is ~500MB

### Slow performance
```bash
# Use GPU if available
# Check: nvidia-smi

# Reduce emotions returned
EMOTION_TOP_K=1

# Increase threshold
EMOTION_THRESHOLD=0.50
```

### Disable if needed
```bash
EMOTION_ENABLE=false
```

## 📚 Full Documentation

See [EMOTION_DETECTION.md](EMOTION_DETECTION.md) for complete details.

## 🎯 Key Files

- `ml_models/emotion_detector_model/preTrainedModel/robertA_model.py` - Model wrapper
- `routes/chat.py` - Integration point
- `tests/test_emotion_integration.py` - Unit tests
- `demo_emotion_detection.py` - Demo script
- `.env.example` - Configuration template

## ✨ Features

✅ Multi-label detection (multiple emotions at once)
✅ 28 emotion categories
✅ Confidence scores (0.00-1.00)
✅ Configurable thresholds
✅ GPU acceleration
✅ Non-blocking (errors don't stop chat)
✅ Detailed logging
✅ Performance tracking
✅ Preserves TTS pipeline

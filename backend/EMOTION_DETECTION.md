# Emotion Detection Integration

## Overview

Isabella now includes emotion-aware message processing using a pretrained RoBERTa-based model (`SamLowe/roberta-base-go_emotions`). The system automatically detects emotions in user messages and uses this information to adjust the AI's tone and response style.

## Features

- **Multi-label emotion detection**: Can detect multiple emotions simultaneously with confidence scores
- **28 emotion categories**: Including joy, sadness, anger, fear, gratitude, excitement, and more
- **Configurable thresholds**: Control sensitivity of emotion detection
- **Non-blocking**: Emotion detection failures never block the chat response
- **Performance tracking**: Logs timing for emotion detection, LLM, and TTS separately
- **GPU acceleration**: Automatically uses CUDA if available, falls back to CPU

## Architecture

### Flow

1. **User sends message** → Chat endpoint receives request
2. **Emotion detection** → Message analyzed for emotional content (if enabled)
3. **System prompt augmentation** → Detected emotions injected into system instruction
4. **LLM generation** → AI generates response with emotion context
5. **TTS generation** → Response converted to speech (unchanged from before)

### Files Modified

- `backend/ml_models/emotion_detector_model/preTrainedModel/robertA_model.py` - Emotion detection model
- `backend/routes/chat.py` - Chat endpoint with emotion integration
- `backend/requirements.txt` - Added torch and transformers dependencies
- `backend/tests/test_emotion_integration.py` - Unit tests

## Configuration

### Environment Variables

Add these to your `.env` file:

```bash
# Enable or disable emotion detection (default: true)
EMOTION_ENABLE=true

# Minimum confidence threshold for emotions (default: 0.35)
EMOTION_THRESHOLD=0.35

# Optional: Limit number of emotions returned (default: no limit)
EMOTION_TOP_K=3
```

### Examples

**High sensitivity** (detect more emotions):
```bash
EMOTION_THRESHOLD=0.25
```

**Low sensitivity** (only strong emotions):
```bash
EMOTION_THRESHOLD=0.50
```

**Disable emotion detection**:
```bash
EMOTION_ENABLE=false
```

**Limit to top 2 emotions**:
```bash
EMOTION_TOP_K=2
```

## Installation

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   This will install:
   - `torch>=2.0.0` - PyTorch for model inference
   - `transformers>=4.30.0` - Hugging Face transformers library

2. **First run**: The model will be automatically downloaded from Hugging Face on first use (~500MB)

3. **GPU support** (optional but recommended):
   - If CUDA is available, the model will use GPU acceleration
   - Otherwise, it runs on CPU (slower but functional)

## Usage

The emotion detection is fully automatic. When a user sends a message:

```python
# User message: "I'm so excited about this!"

# System detects emotions:
# - joy (0.85)
# - excitement (0.72)
# - optimism (0.65)

# System prompt includes:
"""
Current user emotional state (multi-label):
- joy (0.85)
- excitement (0.72)
- optimism (0.65)
Guidelines:
1. Acknowledge the expressed emotions succinctly and naturally.
2. Adjust tone: supportive for distress, encouraging for anxiety/anticipation, celebratory for positive emotions.
3. Do not infer emotions not listed.
4. Preserve all existing system rules and persona instructions.
"""
```

## Logging

The system logs detailed information about emotion detection:

```
📨 NEW CHAT REQUEST
User message: I'm feeling great today!
🎭 Running emotion detection...
✅ Detected emotions (took 0.125s):
   - joy: 0.85
   - optimism: 0.72
🔄 Calling LongCat API...
✓ Received AI response - Length: 156 chars (took 1.234s)
🔊 Generating TTS audio for AI response...
✓ TTS audio generated: speech_abc123.wav (took 2.456s)
⏱️ Performance Summary: Emotion=0.125s | LLM=1.234s | TTS=2.456s | Total=3.815s
```

## Supported Emotions

The model detects 28 different emotions:

**Positive emotions**: admiration, amusement, approval, caring, desire, excitement, gratitude, joy, love, optimism, pride, relief

**Negative emotions**: anger, annoyance, disappointment, disapproval, disgust, embarrassment, fear, grief, nervousness, remorse, sadness

**Neutral/Ambiguous**: confusion, curiosity, realization, surprise, neutral

## Error Handling

- **Model loading fails**: System logs error and continues without emotion detection
- **Prediction fails**: System logs warning and continues with standard LLM generation
- **Empty/whitespace message**: Emotion detection is skipped
- **Network issues**: Only affects model download on first run; gracefully handles failures

## Performance

Typical timings on different hardware:

- **GPU (CUDA)**: 50-150ms per message
- **CPU**: 200-500ms per message
- **Model loading**: 2-5 seconds (one-time, at startup)

The emotion detection runs before the LLM call, adding minimal latency to the overall response time.

## Testing

Run the unit tests:
```bash
cd backend
python -m unittest tests.test_emotion_integration -v
```

Run the demo (no model required):
```bash
cd backend
python demo_emotion_detection.py
```

## API Response

The API response structure remains unchanged:

```json
{
  "reply": "I'm so glad to hear you're feeling great! Your positive energy is wonderful...",
  "audio_file": "speech_abc123.wav"
}
```

(Optional extension: Could add `emotions` field with detected emotions if frontend needs it)

## Troubleshooting

### Model won't download
- Check internet connection
- Ensure Hugging Face is accessible
- May need to set `HF_HOME` environment variable for custom cache location

### Out of memory
- Reduce `EMOTION_TOP_K` to limit processing
- Use CPU instead of GPU (set `CUDA_VISIBLE_DEVICES=""`)
- Increase system RAM or use smaller model variant

### Slow performance
- Enable GPU acceleration (install CUDA)
- Increase `EMOTION_THRESHOLD` to reduce processing
- Set `EMOTION_TOP_K=1` for fastest mode

### False emotion detection
- Increase `EMOTION_THRESHOLD` for higher confidence
- Review emotion labels in logs to understand model behavior
- Consider disabling for certain message types (e.g., commands)

## Future Enhancements

Possible improvements:
- Emotion history tracking across conversation
- Emotion-based response templates
- Adaptive threshold based on conversation context
- Custom emotion mappings for specific use cases
- Multi-language emotion detection
- Batch processing for performance optimization

## References

- Model: [SamLowe/roberta-base-go_emotions](https://huggingface.co/SamLowe/roberta-base-go_emotions)
- Paper: [GoEmotions: A Dataset of Fine-Grained Emotions](https://arxiv.org/abs/2005.00547)
- Framework: [Hugging Face Transformers](https://huggingface.co/docs/transformers)

# Emotion Detection Architecture

## System Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                        USER SENDS MESSAGE                            │
│                     "I'm so excited today!"                          │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     CHAT ENDPOINT (routes/chat.py)                   │
│  • Receives message from frontend                                    │
│  • Logs: "📨 NEW CHAT REQUEST"                                      │
│  • Logs: User message                                                │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │   Check if enabled   │
                    │  EMOTION_ENABLE?     │
                    └──────────┬───────────┘
                              │
                 ┌────────────┴────────────┐
                 │ YES                     │ NO (skip emotion)
                 ▼                         ▼
┌────────────────────────────────────┐   │
│   EMOTION DETECTION                │   │
│   (robertA_model.py)               │   │
│                                    │   │
│  1. Load model (lazy, once)        │   │
│  2. Tokenize input                 │   │
│  3. Run inference (GPU/CPU)        │   │
│  4. Apply sigmoid                  │   │
│  5. Filter by threshold            │   │
│  6. Sort by confidence             │   │
│  7. Apply top_k limit              │   │
│                                    │   │
│  ⏱️ Time taken: ~50-500ms          │   │
│                                    │   │
│  Output:                           │   │
│  [('joy', 0.85),                   │   │
│   ('excitement', 0.78),            │   │
│   ('optimism', 0.62)]              │   │
└────────────────┬───────────────────┘   │
                 │                        │
                 ▼                        │
┌────────────────────────────────────────┴──────────────────────────┐
│   LOG DETECTED EMOTIONS                                            │
│   🎭 Running emotion detection...                                 │
│   ✅ Detected emotions (took 0.125s):                             │
│      - joy: 0.85                                                   │
│      - excitement: 0.78                                            │
│      - optimism: 0.62                                              │
└────────────────────────────┬───────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│   SYSTEM PROMPT AUGMENTATION                                         │
│                                                                      │
│   Base system content:                                               │
│   "You are Isabella/bella. This is a conversation..."                │
│                                                                      │
│   + Emotion context (if emotions detected):                          │
│   """                                                                │
│   Current user emotional state (multi-label):                        │
│   - joy (0.85)                                                       │
│   - excitement (0.78)                                                │
│   - optimism (0.62)                                                  │
│   Guidelines:                                                        │
│   1. Acknowledge the expressed emotions succinctly and naturally.    │
│   2. Adjust tone: supportive for distress, encouraging for           │
│      anxiety/anticipation, celebratory for positive emotions.        │
│   3. Do not infer emotions not listed.                               │
│   4. Preserve all existing system rules and persona instructions.    │
│   """                                                                │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│   SAVE USER MESSAGE TO DATABASE                                      │
│   (ChatService.save_message)                                         │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│   GET CONTEXT MESSAGES                                               │
│   (Last 10 messages for conversation context)                        │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│   BUILD MESSAGES ARRAY                                               │
│   [                                                                  │
│     {role: "system", content: augmented_system_content},            │
│     {role: "user", content: original_message}                       │
│   ]                                                                  │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│   CALL LLM API (LongCat)                                            │
│   🔄 Calling LongCat API...                                         │
│   ⏱️ Time taken: ~1-3 seconds                                       │
│                                                                      │
│   LLM response includes emotion-aware content:                       │
│   "That's wonderful! Your excitement is contagious! 🎉..."          │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│   LOG LLM RESPONSE                                                   │
│   ✓ Received AI response - Length: 156 chars (took 1.234s)         │
│   AI response preview: That's wonderful! Your excitement is...       │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│   SAVE AI RESPONSE TO DATABASE                                       │
│   (ChatService.save_message)                                         │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│   GENERATE TTS AUDIO (UNCHANGED)                                     │
│   🔊 Generating TTS audio for AI response...                        │
│   • Clean text (remove markdown, emojis)                            │
│   • Call Piper TTS                                                   │
│   • Generate audio file                                              │
│   ⏱️ Time taken: ~1-3 seconds                                       │
│                                                                      │
│   ✓ TTS audio generated: speech_abc123.wav (took 2.456s)           │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│   LOG PERFORMANCE SUMMARY                                            │
│   ⏱️ Performance Summary:                                           │
│      Emotion=0.125s | LLM=1.234s | TTS=2.456s | Total=3.815s       │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│   RETURN RESPONSE                                                    │
│   {                                                                  │
│     "reply": "That's wonderful! Your excitement is contagious! 🎉",│
│     "audio_file": "speech_abc123.wav"                               │
│   }                                                                  │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  FRONTEND SHOWS  │
                    │  TEXT & PLAYS    │
                    │     AUDIO        │
                    └─────────────────┘
```

## Component Details

### 1. Emotion Detector Model (robertA_model.py)

**Class**: `EmotionDetectorModel`
- **Pattern**: Lazy Singleton
- **Model**: SamLowe/roberta-base-go_emotions
- **Device**: Auto-detects CUDA/CPU
- **Load Time**: 2-5 seconds (one-time)

**Function**: `predict_emotions(text, threshold, top_k)`
- **Input**: User text string
- **Output**: List of (label, confidence) tuples
- **Processing**:
  1. Tokenize (max 512 tokens)
  2. Model inference
  3. Sigmoid activation (multi-label)
  4. Filter by threshold
  5. Sort descending
  6. Apply top_k

**Function**: `format_emotions(emotions)`
- **Input**: List of (label, confidence)
- **Output**: Formatted string for prompt
- **Format**: `- emotion (0.XX)`

### 2. Chat Route (routes/chat.py)

**Modifications**:
- Import emotion detection functions
- Read environment configuration
- Call predict_emotions() before LLM
- Build augmented system prompt
- Add timing for all stages
- Log performance summary

**Configuration Variables**:
- `EMOTION_ENABLE` (bool, default: true)
- `EMOTION_THRESHOLD` (float, default: 0.35)
- `EMOTION_TOP_K` (int, optional)

### 3. System Prompt Template

**Structure**:
```
Base persona + Context window
+
Emotion augmentation (if emotions detected):
  Current user emotional state (multi-label):
  {formatted_emotions}
  Guidelines:
  1. Acknowledge emotions naturally
  2. Adjust tone appropriately
  3. Don't infer unlisted emotions
  4. Preserve existing rules
```

## Data Flow

### Request Path
```
Frontend → Chat Route → Emotion Detection → System Prompt → LLM → TTS → Response
```

### Timing Breakdown
```
Total:   3.815s
├─ Emotion:  0.125s (3.3%)
├─ LLM:      1.234s (32.3%)
└─ TTS:      2.456s (64.4%)
```

## Error Handling

### Emotion Detection Failure
```python
try:
    emotions = predict_emotions(text)
except Exception as e:
    logger.warning(f"Emotion detection failed: {e}")
    emotions = []  # Continue without emotions
```

**Result**: Chat flow continues normally without emotion augmentation

### Model Load Failure
```python
try:
    model = EmotionDetectorModel()
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    # Model stays None, predict_emotions returns []
```

**Result**: System starts but skips emotion detection

## Configuration Scenarios

### 1. Default (Enabled)
```bash
EMOTION_ENABLE=true
EMOTION_THRESHOLD=0.35
```
- Detects emotions above 35% confidence
- Returns all emotions above threshold
- Normal operation

### 2. Disabled
```bash
EMOTION_ENABLE=false
```
- Skips emotion detection entirely
- No model loading
- System prompt unchanged

### 3. High Sensitivity
```bash
EMOTION_ENABLE=true
EMOTION_THRESHOLD=0.25
EMOTION_TOP_K=5
```
- Detects more emotions (lower threshold)
- Limited to top 5 emotions
- More emotion context in prompt

### 4. Low Sensitivity
```bash
EMOTION_ENABLE=true
EMOTION_THRESHOLD=0.50
EMOTION_TOP_K=2
```
- Only strong emotions
- Limited to top 2
- Minimal emotion context

## Performance Optimization

### Lazy Loading
- Model loaded only on first prediction
- Singleton pattern prevents reloading
- Startup time not affected

### GPU Acceleration
```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```
- Automatic GPU detection
- 3-5x faster on GPU
- Graceful CPU fallback

### Caching
- Model tokenizer cached
- Model weights cached
- No per-request downloads

## Security Considerations

✅ **Input Validation**: Empty/whitespace handled
✅ **Error Handling**: All exceptions caught
✅ **Resource Limits**: Max 512 tokens
✅ **No Code Execution**: Pure inference
✅ **Logging**: No sensitive data logged

## Testing Strategy

### Unit Tests (test_emotion_integration.py)
- Test emotion formatting
- Test empty input handling
- Test configuration parameters
- Test error handling
- Mock model for testing without dependencies

### Integration Test (demo_emotion_detection.py)
- Shows complete flow
- Demonstrates all emotion categories
- No model required (mocked)
- Visual output for verification

### Manual Testing
```bash
# Start server
uvicorn main:app --reload

# Send test message
curl -X POST localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "I am excited!", "thinking": false}'

# Check logs for emotion detection
```

## Deployment Checklist

- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Configure environment: Copy `.env.example` to `.env`
- [ ] Set LONGCAT_API_KEY
- [ ] Optional: Configure EMOTION_* variables
- [ ] Start server: `uvicorn main:app`
- [ ] Verify model loads (check logs for "Loading emotion detection model")
- [ ] Test with sample message
- [ ] Monitor performance (check timing logs)

## Troubleshooting

### Issue: Model not loading
**Check**: Internet connection, disk space (~500MB)
**Solution**: Model downloads from Hugging Face on first run

### Issue: Slow performance
**Check**: GPU availability
**Solution**: Install CUDA or increase threshold/top_k

### Issue: Out of memory
**Check**: System RAM, GPU memory
**Solution**: Use CPU mode or increase system resources

### Issue: Incorrect emotions
**Check**: Threshold settings, message content
**Solution**: Adjust EMOTION_THRESHOLD or review expected behavior

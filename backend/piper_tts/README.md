# Piper TTS Setup

This directory should contain:

1. **piper/** folder - Contains the Piper TTS binary
   - Download from: https://github.com/rhasspy/piper/releases
   - Place the `piper` executable in this folder

2. **Voice model files** (place in the piper_tts directory):
   - `en_US-amy-medium.onnx` - Voice model
   - `en_US-amy-medium.onnx.json` - Voice configuration

## Setup Instructions:

1. Download Piper TTS binary for your platform from: https://github.com/rhasspy/piper/releases
2. Extract and place the `piper` executable in the `piper/` folder
3. Download the en_US-amy-medium voice model from: https://github.com/rhasspy/piper/releases/tag/2023.11.14-2
4. Place both `en_US-amy-medium.onnx` and `en_US-amy-medium.onnx.json` in this directory

## Directory Structure:
```
piper_tts/
├── piper/
│   └── piper (executable)
├── en_US-amy-medium.onnx
├── en_US-amy-medium.onnx.json
└── README.md
```

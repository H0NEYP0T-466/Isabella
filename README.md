# Isabella

<p align="center">

  <!-- Core -->
  ![GitHub License](https://img.shields.io/github/license/H0NEYP0T-466/Isabella?style=for-the-badge&color=brightgreen)
  ![GitHub Stars](https://img.shields.io/github/stars/H0NEYP0T-466/Isabella?style=for-the-badge&color=yellow)
  ![GitHub Forks](https://img.shields.io/github/forks/H0NEYP0T-466/Isabella?style=for-the-badge&color=blue)
  ![GitHub Issues](https://img.shields.io/github/issues/H0NEYP0T-466/Isabella?style=for-the-badge&color=red)
  ![GitHub Pull Requests](https://img.shields.io/github/issues-pr/H0NEYP0T-466/Isabella?style=for-the-badge&color=orange)
  ![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-brightgreen?style=for-the-badge)

  <!-- Activity -->
  ![Last Commit](https://img.shields.io/github/last-commit/H0NEYP0T-466/Isabella?style=for-the-badge&color=purple)
  ![Commit Activity](https://img.shields.io/github/commit-activity/m/H0NEYP0T-466/Isabella?style=for-the-badge&color=teal)
  ![Repo Size](https://img.shields.io/github/repo-size/H0NEYP0T-466/Isabella?style=for-the-badge&color=blueviolet)
  ![Code Size](https://img.shields.io/github/languages/code-size/H0NEYP0T-466/Isabella?style=for-the-badge&color=indigo)

  <!-- Languages -->
  ![Top Language](https://img.shields.io/github/languages/top/H0NEYP0T-466/Isabella?style=for-the-badge&color=critical)
  ![Languages Count](https://img.shields.io/github/languages/count/H0NEYP0T-466/Isabella?style=for-the-badge&color=success)

  <!-- Community -->
  ![Documentation](https://img.shields.io/badge/Docs-Available-green?style=for-the-badge&logo=readthedocs&logoColor=white)
  ![Open Source Love](https://img.shields.io/badge/Open%20Source-%E2%9D%A4-red?style=for-the-badge)

</p>

## 📝 About

Isabella is an AI-powered chatbot with a terminal-style UI, featuring emotion detection capabilities. Built with React + TypeScript frontend and FastAPI backend, powered by LongCat API and advanced ML models for emotion analysis.

## 🔗 Quick Links

- [**Demo**](#-usage)
- [**Documentation**](#-table-of-contents)
- [**Issues**](https://github.com/H0NEYP0T-466/Isabella/issues)
- [**Contributing**](CONTRIBUTING.md)
- [**Security**](SECURITY.md)

## 📑 Table of Contents

- [About](#-about)
- [Quick Links](#-quick-links)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Dependencies & Packages](#-dependencies--packages)
- [Installation](#-installation)
- [Usage](#-usage)
- [API Endpoints](#-api-endpoints)
- [Folder Structure](#-folder-structure)
- [Development](#-development)
- [Environment Variables](#-environment-variables)
- [MongoDB Configuration](#-mongodb-configuration)
- [Logging](#-logging)
- [Contributing](#-contributing)
- [License](#-license)
- [Security](#-security)
- [Code of Conduct](#-code-of-conduct)

## ✨ Features

- 🤖 **AI-Powered Chat**: Leverages LongCat API for intelligent conversations
- 🎨 **Terminal Aesthetic**: Black background, green text, monospace font for authentic CLI feel
- 🧠 **Thinking Mode Toggle**:
  - ON: Uses `LongCat-Thinker` model (deeper reasoning)
  - OFF: Uses `LongCat-Flash-Chat` model (faster responses)
- 😊 **Emotion Detection**: Advanced ML-based emotion analysis using PyTorch and Transformers
- 💾 **MongoDB Integration**: Persistent chat history storage
- 📜 **Chat History**: Loads last 50 messages on startup
- 🔄 **Context Window**: Sends last 10 messages to AI for conversation continuity
- 📊 **Comprehensive Logging**: Detailed server-side logs for all operations
- 📜 **Auto-scroll**: Chat window automatically scrolls to show new messages
- 🔊 **Text-to-Speech**: AI responses spoken using Piper TTS (local, offline)
- ⚡ **Single-page Application**: No routing, streamlined UX
- 🔒 **Type-safe Implementation**: Full TypeScript for frontend reliability

## 🛠 Tech Stack

### Languages
![TypeScript](https://img.shields.io/badge/TypeScript-%23007ACC.svg?style=for-the-badge&logo=typescript&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![JavaScript](https://img.shields.io/badge/JavaScript-%23323330.svg?style=for-the-badge&logo=javascript&logoColor=%23F7DF1E)

### Frameworks & Libraries
![React](https://img.shields.io/badge/React-19-%2361DAFB.svg?style=for-the-badge&logo=react&logoColor=black)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Vite](https://img.shields.io/badge/Vite-%23646CFF.svg?style=for-the-badge&logo=vite&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=pytorch&logoColor=white)

### Databases
![MongoDB](https://img.shields.io/badge/MongoDB-%234ea94b.svg?style=for-the-badge&logo=mongodb&logoColor=white)

### DevOps / CI / Tools
![ESLint](https://img.shields.io/badge/ESLint-4B3263?style=for-the-badge&logo=eslint&logoColor=white)
![npm](https://img.shields.io/badge/npm-%23CB3837.svg?style=for-the-badge&logo=npm&logoColor=white)
![Git](https://img.shields.io/badge/Git-%23F05033.svg?style=for-the-badge&logo=git&logoColor=white)

## 📦 Dependencies & Packages

### Frontend Dependencies

#### Runtime Dependencies

[![axios](https://img.shields.io/npm/v/axios?style=for-the-badge&label=axios&color=blue)](https://www.npmjs.com/package/axios)
[![react](https://img.shields.io/npm/v/react?style=for-the-badge&label=react&color=blue)](https://www.npmjs.com/package/react)
[![react-dom](https://img.shields.io/npm/v/react-dom?style=for-the-badge&label=react-dom&color=blue)](https://www.npmjs.com/package/react-dom)
[![react-markdown](https://img.shields.io/npm/v/react-markdown?style=for-the-badge&label=react-markdown&color=blue)](https://www.npmjs.com/package/react-markdown)

- **axios** `^1.13.2` - Promise-based HTTP client for API requests
- **react** `^19.2.0` - Core React library
- **react-dom** `^19.2.0` - React DOM rendering
- **react-markdown** `^10.1.0` - Markdown rendering in React

<details>
<summary><strong>Dev/Build/Test Dependencies</strong></summary>

[![@eslint/js](https://img.shields.io/npm/v/@eslint/js?style=for-the-badge&label=%40eslint%2Fjs&color=purple)](https://www.npmjs.com/package/@eslint/js)
[![@types/node](https://img.shields.io/npm/v/@types/node?style=for-the-badge&label=%40types%2Fnode&color=purple)](https://www.npmjs.com/package/@types/node)
[![@types/react](https://img.shields.io/npm/v/@types/react?style=for-the-badge&label=%40types%2Freact&color=purple)](https://www.npmjs.com/package/@types/react)
[![@types/react-dom](https://img.shields.io/npm/v/@types/react-dom?style=for-the-badge&label=%40types%2Freact-dom&color=purple)](https://www.npmjs.com/package/@types/react-dom)
[![@vitejs/plugin-react](https://img.shields.io/npm/v/@vitejs/plugin-react?style=for-the-badge&label=%40vitejs%2Fplugin-react&color=purple)](https://www.npmjs.com/package/@vitejs/plugin-react)
[![eslint](https://img.shields.io/npm/v/eslint?style=for-the-badge&label=eslint&color=purple)](https://www.npmjs.com/package/eslint)
[![eslint-plugin-react-hooks](https://img.shields.io/npm/v/eslint-plugin-react-hooks?style=for-the-badge&label=eslint-plugin-react-hooks&color=purple)](https://www.npmjs.com/package/eslint-plugin-react-hooks)
[![eslint-plugin-react-refresh](https://img.shields.io/npm/v/eslint-plugin-react-refresh?style=for-the-badge&label=eslint-plugin-react-refresh&color=purple)](https://www.npmjs.com/package/eslint-plugin-react-refresh)
[![globals](https://img.shields.io/npm/v/globals?style=for-the-badge&label=globals&color=purple)](https://www.npmjs.com/package/globals)
[![typescript](https://img.shields.io/npm/v/typescript?style=for-the-badge&label=typescript&color=purple)](https://www.npmjs.com/package/typescript)
[![typescript-eslint](https://img.shields.io/npm/v/typescript-eslint?style=for-the-badge&label=typescript-eslint&color=purple)](https://www.npmjs.com/package/typescript-eslint)
[![vite](https://img.shields.io/npm/v/vite?style=for-the-badge&label=vite&color=purple)](https://www.npmjs.com/package/vite)

- **@eslint/js** `^9.39.1` - ESLint JavaScript configuration
- **@types/node** `^24.10.0` - TypeScript definitions for Node.js
- **@types/react** `^19.2.2` - TypeScript definitions for React
- **@types/react-dom** `^19.2.2` - TypeScript definitions for React DOM
- **@vitejs/plugin-react** `^5.1.0` - Vite plugin for React
- **eslint** `^9.39.1` - JavaScript/TypeScript linter
- **eslint-plugin-react-hooks** `^5.2.0` - ESLint rules for React Hooks
- **eslint-plugin-react-refresh** `^0.4.24` - ESLint plugin for React Fast Refresh
- **globals** `^16.5.0` - Global identifiers from different JavaScript environments
- **typescript** `~5.9.3` - TypeScript compiler
- **typescript-eslint** `^8.46.3` - TypeScript ESLint parser and plugin
- **vite** `^7.2.2` - Next-generation frontend build tool

</details>

### Backend Dependencies

#### Runtime Dependencies

[![fastapi](https://img.shields.io/pypi/v/fastapi?style=for-the-badge&label=fastapi&color=green)](https://pypi.org/project/fastapi/)
[![uvicorn](https://img.shields.io/pypi/v/uvicorn?style=for-the-badge&label=uvicorn&color=green)](https://pypi.org/project/uvicorn/)
[![httpx](https://img.shields.io/pypi/v/httpx?style=for-the-badge&label=httpx&color=green)](https://pypi.org/project/httpx/)
[![python-dotenv](https://img.shields.io/pypi/v/python-dotenv?style=for-the-badge&label=python-dotenv&color=green)](https://pypi.org/project/python-dotenv/)
[![motor](https://img.shields.io/pypi/v/motor?style=for-the-badge&label=motor&color=green)](https://pypi.org/project/motor/)
[![pymongo](https://img.shields.io/pypi/v/pymongo?style=for-the-badge&label=pymongo&color=green)](https://pypi.org/project/pymongo/)
[![torch](https://img.shields.io/pypi/v/torch?style=for-the-badge&label=torch&color=green)](https://pypi.org/project/torch/)
[![transformers](https://img.shields.io/pypi/v/transformers?style=for-the-badge&label=transformers&color=green)](https://pypi.org/project/transformers/)

- **fastapi** `0.115.0` - Modern, fast web framework for building APIs
- **uvicorn** `0.32.0` - ASGI server implementation
- **httpx** `0.27.2` - Async HTTP client
- **python-dotenv** `1.0.1` - Environment variable management
- **motor** `3.3.2` - Async MongoDB driver
- **pymongo** `4.6.1` - MongoDB driver for Python
- **torch** `>=2.0.0` - PyTorch machine learning framework
- **transformers** `>=4.30.0` - Hugging Face transformers for NLP/ML

## 🚀 Installation

### Prerequisites

- **Node.js** 18+ and npm
- **Python** 3.8+
- **MongoDB** 7.0+ (locally or via Docker)
- **Git**

### Backend Setup

1. **Install and start MongoDB:**
   ```bash
   # Using Docker (recommended)
   docker run -d -p 27017:27017 --name mongodb mongo:7.0
   
   # Or install MongoDB locally and start it
   # mongod --dbpath /path/to/data
   ```

2. Navigate to the backend directory:
   ```bash
   cd backend
   ```

3. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Create a `.env` file with your LongCat API key:
   ```bash
   echo "LONGCAT_API_KEY=your_actual_api_key_here" > .env
   ```

6. **Setup Piper TTS (Optional - for text-to-speech):**
   
   a. Download Piper TTS binary for your platform:
      - Visit: https://github.com/rhasspy/piper/releases
      - Download the appropriate version for your OS
      - Extract and place the `piper` executable in `backend/piper_tts/piper/`
   
   b. Download the en_US-amy-medium voice model:
      - Visit: https://github.com/rhasspy/piper/releases/tag/2023.11.14-2
      - Download `en_US-amy-medium.onnx` and `en_US-amy-medium.onnx.json`
      - Place both files in `backend/piper_tts/`
   
   See `backend/piper_tts/README.md` for detailed instructions.
   
   **Note:** TTS is optional. The chatbot will work without it.

7. Start the FastAPI server:
   ```bash
   uvicorn main:app --reload --port 5000
   ```

   The backend will run at: `http://localhost:5000`

### Frontend Setup

1. Install dependencies:
   ```bash
   npm install
   ```

2. Start the development server:
   ```bash
   npm run dev
   ```

   The frontend will run at: `http://localhost:5173`

## ⚡ Usage

1. Open the frontend in your browser (`http://localhost:5173`)
2. You'll see a terminal-style interface with:
   - A "Thinking Mode" checkbox at the top
   - A chat window showing conversation history
   - An input box at the bottom for typing messages
3. Toggle "Thinking Mode" to switch between AI models:
   - ✅ ON: Uses LongCat-Thinker (thoughtful, detailed responses)
   - ⬜ OFF: Uses LongCat-Flash-Chat (faster, concise responses)
4. Type your message and press Enter or click SEND
5. The AI response will appear in the terminal window
6. The chat window will automatically scroll to show new messages
7. If TTS is configured, AI responses will be spoken automatically
8. Audio controls appear below each AI message for manual playback

## 📡 API Endpoints

### POST `/chat`
Send a message to the AI chatbot.

**Request Body:**
```json
{
  "message": "Your question here",
  "thinking": true
}
```

**Response:**
```json
{
  "reply": "AI response here",
  "audio_file": "speech_uuid.wav"
}
```

### GET `/messages`
Fetch the last 50 messages from chat history.

**Response:**
```json
{
  "messages": [
    {
      "_id": "...",
      "role": "user",
      "content": "Hello!",
      "timestamp": "2025-11-10T14:02:31.537000",
      "thinking": false,
      "model": "LongCat-Flash-Chat"
    }
  ]
}
```

### POST `/tts`
Generate speech from text using Piper TTS.

**Request Body:**
```json
{
  "text": "Text to convert to speech"
}
```

**Response:**
```json
{
  "audio_file": "speech_uuid.wav"
}
```

### GET `/tts/audio/{filename}`
Retrieve a generated audio file.

**Response:**
- Audio file in WAV format

## 📂 Folder Structure

```
Isabella/
├── src/                    # Frontend React application
│   ├── components/
│   │   ├── ChatWindow.tsx
│   │   ├── ThinkingToggle.tsx
│   │   └── IsolateToggle.tsx
│   ├── assets/
│   ├── App.tsx
│   ├── App.css
│   ├── main.tsx
│   └── index.css
├── backend/
│   ├── config/            # Configuration modules
│   │   └── database.py    # MongoDB connection
│   ├── models/            # Data models
│   │   └── chat.py
│   ├── routes/            # API routes
│   │   ├── chat.py
│   │   └── tts.py
│   ├── services/          # Business logic
│   │   ├── chat_service.py
│   │   └── tts_service.py
│   ├── ml_models/         # Machine learning models
│   │   └── emotion_detector_model/
│   ├── datasets/          # Training datasets
│   │   └── emotion_detection_dataset/
│   ├── tests/             # Backend tests
│   │   ├── test_emotion_integration.py
│   │   └── test_timestamp_context.py
│   ├── utils/             # Utilities
│   │   └── logger.py
│   ├── main.py            # FastAPI entry point
│   ├── requirements.txt
│   ├── ARCHITECTURE.md
│   ├── EMOTION_DETECTION.md
│   ├── QUICKSTART_EMOTION.md
│   └── README.md
├── public/
│   └── vite.svg
├── .github/               # GitHub configuration
│   ├── ISSUE_TEMPLATE/    # Issue templates
│   └── pull_request_template.md
├── package.json
├── package-lock.json
├── tsconfig.json
├── tsconfig.app.json
├── tsconfig.node.json
├── vite.config.ts
├── eslint.config.js
├── index.html
├── README.md
├── LICENSE
├── CONTRIBUTING.md
├── SECURITY.md
└── CODE_OF_CONDUCT.md
```

## 💻 Development

### Build Frontend
```bash
npm run build
```

### Lint Frontend
```bash
npm run lint
```

### Preview Production Build
```bash
npm run preview
```

### Run Backend Tests
```bash
cd backend
python -m pytest tests/
```

## 🔐 Environment Variables

### Backend `.env`
- `LONGCAT_API_KEY`: Your LongCat API key (required)

## 🗄 MongoDB Configuration

The application uses MongoDB to store chat history:
- **Connection URL**: `mongodb://127.0.0.1:27017/isabella`
- **Database**: `isabella`
- **Collection**: `chats`

### Database Schema
```javascript
{
  "_id": ObjectId,
  "role": String,          // "user" or "assistant"
  "content": String,       // Message content
  "timestamp": ISODate,    // Message timestamp
  "thinking": Boolean,     // Thinking mode enabled
  "model": String          // AI model used
}
```

## 📊 Logging

The backend provides comprehensive logging for debugging and monitoring:
- MongoDB connection status
- All user messages and AI responses
- Context window contents (last 10 messages sent to AI)
- API calls and errors
- Database operations
- Emotion detection results

Check the server console for detailed logs of all operations.

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details on:
- How to fork and contribute
- Code style and linting rules
- Bug reporting and feature requests
- Testing and documentation

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🛡 Security

Security is important to us. Please see our [Security Policy](SECURITY.md) for information on:
- Reporting vulnerabilities
- Security contact information
- Vulnerability handling process

## 📏 Code of Conduct

This project adheres to the Contributor Covenant [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## 📝 Notes

- The backend must be running on port 5000 for the frontend to connect properly
- MongoDB must be running on port 27017 (default)
- Update the API URL in `App.tsx` if deploying to production
- For production use, configure CORS properly in `main.py` with specific allowed origins
- The terminal styling uses monospace fonts and green (#0f0) text on black (#111) background
- Chat history is automatically loaded when the page loads
- The AI receives the last 10 messages as context for better conversation continuity

---

<p align="center">Made with ❤ by H0NEYP0T-466</p>

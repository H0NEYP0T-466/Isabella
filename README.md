# Isabella - AI Chatbot with LongCat API

A simple AI chatbot with terminal-style UI built with React + TypeScript frontend and FastAPI backend, powered by LongCat API.

## Features

- 🤖 AI-powered chat using LongCat API
- 🎨 Terminal/CLI aesthetic (black background, green text, monospace font)
- 🧠 Thinking Mode toggle:
  - ON: Uses `LongCat-Thinker` model (deeper reasoning)
  - OFF: Uses `LongCat-Flash-Chat` model (faster responses)
- 💾 **MongoDB Integration**: Persistent chat history storage
- 📜 **Chat History**: Loads last 50 messages on startup
- 🔄 **Context Window**: Sends last 10 messages to AI for conversation continuity
- 📊 **Comprehensive Logging**: Detailed server-side logs for all operations
- ⚡ Single-page application (no routing)
- 🔒 Type-safe TypeScript implementation

## Tech Stack

### Frontend
- React 19 + TypeScript
- Vite (build tool)
- Axios (HTTP client)
- Terminal-style CSS

### Backend
- FastAPI (Python web framework)
- HTTPX (async HTTP client)
- Python-dotenv (environment variables)
- Motor (async MongoDB driver)
- PyMongo (MongoDB driver)

## Project Structure

```
Isabella/
├── src/                    # Frontend React application
│   ├── components/
│   │   ├── ChatWindow.tsx
│   │   └── ThinkingToggle.tsx
│   ├── App.tsx
│   ├── main.tsx
│   └── index.css
├── backend/
│   ├── config/            # Configuration modules
│   │   └── database.py    # MongoDB connection
│   ├── models/            # Data models
│   │   └── chat.py
│   ├── routes/            # API routes
│   │   └── chat.py
│   ├── services/          # Business logic
│   │   └── chat_service.py
│   ├── utils/             # Utilities
│   │   └── logger.py
│   ├── main.py            # FastAPI entry point
│   ├── requirements.txt
│   └── README.md          # Backend documentation
├── package.json
└── README.md
```

## Setup Instructions

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

6. Start the FastAPI server:
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

   The frontend will run at: `http://localhost:5173` (or another port if 5173 is busy)

## Usage

1. Open the frontend in your browser (e.g., `http://localhost:5173`)
2. You'll see a terminal-style interface with:
   - A "Thinking Mode" checkbox at the top
   - A chat window showing conversation history
   - An input box at the bottom for typing messages
3. Toggle "Thinking Mode" to switch between AI models:
   - ✅ ON: Uses LongCat-Thinker (more thoughtful, detailed responses)
   - ⬜ OFF: Uses LongCat-Flash-Chat (faster, concise responses)
4. Type your message and press Enter or click SEND
5. The AI response will appear in the terminal window

## API Endpoints

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
  "reply": "AI response here"
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

## Development

### Build Frontend
```bash
npm run build
```

### Lint Frontend
```bash
npm run lint
```

### Run Tests (if added)
```bash
npm test
```

## Environment Variables

### Backend `.env`
- `LONGCAT_API_KEY`: Your LongCat API key (required)

## MongoDB Configuration

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

## Logging

The backend provides comprehensive logging for debugging and monitoring:
- MongoDB connection status
- All user messages and AI responses
- Context window contents (last 10 messages sent to AI)
- API calls and errors
- Database operations

Check the server console for detailed logs of all operations.

## Notes

- The backend must be running on port 5000 for the frontend to connect properly
- MongoDB must be running on port 27017 (default)
- Update the API URL in `App.tsx` if deploying to production
- For production use, configure CORS properly in `main.py` with specific allowed origins
- The terminal styling uses monospace fonts and green (#0f0) text on black (#111) background
- Chat history is automatically loaded when the page loads
- The AI receives the last 10 messages as context for better conversation continuity

## License

This project is open source and available under the MIT License.

# Isabella Backend

FastAPI-based backend for Isabella AI chatbot with MongoDB integration.

## Project Structure

```
backend/
├── config/          # Configuration modules
│   └── database.py  # MongoDB connection management
├── models/          # Data models
│   └── chat.py      # Chat message models
├── routes/          # API routes
│   └── chat.py      # Chat endpoints
├── services/        # Business logic layer
│   └── chat_service.py  # Chat operations (CRUD)
├── utils/           # Utility functions
│   └── logger.py    # Logging configuration
├── main.py          # FastAPI application entry point
└── requirements.txt # Python dependencies

```

## Features

- **MongoDB Integration**: Stores all chat history in MongoDB
- **Context Window**: Sends last 10 messages as context to AI
- **Chat History**: Fetches and returns last 50 messages
- **Comprehensive Logging**: Detailed server-side logs for all operations
- **Clean Architecture**: Organized folder structure with separation of concerns

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create `.env` file:
```
LONGCAT_API_KEY=your_api_key_here
```

3. Start MongoDB:
```bash
# Using Docker
docker run -d -p 27017:27017 --name mongodb mongo:7.0

# Or use local MongoDB
mongod --dbpath /path/to/data
```

4. Run the server:
```bash
uvicorn main:app --host 0.0.0.0 --port 5000 --reload
```

## API Endpoints

### POST /chat
Send a chat message and get AI response.

**Request:**
```json
{
  "message": "Hello, Isabella!",
  "thinking": false
}
```

**Response:**
```json
{
  "reply": "Hello! How can I help you today?"
}
```

### GET /messages
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

### GET /
Health check endpoint.

## Database Schema

### Collection: `chats`

```javascript
{
  "_id": ObjectId,
  "role": String,          // "user" or "assistant"
  "content": String,       // Message content
  "timestamp": ISODate,    // Message timestamp
  "thinking": Boolean,     // Whether thinking mode was enabled
  "model": String          // AI model used
}
```

## Logging

The backend provides comprehensive logging for:
- MongoDB connection status
- User messages and AI responses
- Context window contents
- API calls and errors
- All database operations

Example log output:
```
2025-11-10 14:02:31 - root - INFO - 📨 NEW CHAT REQUEST
2025-11-10 14:02:31 - root - INFO - User message: Hello Isabella!
2025-11-10 14:02:31 - services.chat_service - INFO - ✓ Message saved to database
2025-11-10 14:02:31 - services.chat_service - INFO - ✓ Retrieved 5 messages for context window
```

## MongoDB Connection

The application connects to MongoDB at:
```
mongodb://127.0.0.1:27017/isabella
```

Database: `isabella`
Collection: `chats`

## Environment Variables

- `LONGCAT_API_KEY`: API key for LongCat AI service (required)

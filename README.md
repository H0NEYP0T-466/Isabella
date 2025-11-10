# Isabella - AI Chatbot with LongCat API

A simple AI chatbot with terminal-style UI built with React + TypeScript frontend and FastAPI backend, powered by LongCat API.

## Features

- 🤖 AI-powered chat using LongCat API
- 🎨 Terminal/CLI aesthetic (black background, green text, monospace font)
- 🧠 Thinking Mode toggle:
  - ON: Uses `LongCat-Thinker` model (deeper reasoning)
  - OFF: Uses `LongCat-Flash-Chat` model (faster responses)
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

## Project Structure

```
Isabella/
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── ChatWindow.tsx
│   │   │   └── ThinkingToggle.tsx
│   │   ├── App.tsx
│   │   ├── main.tsx
│   │   └── index.css
│   └── package.json
├── backend/
│   ├── main.py
│   ├── requirements.txt
│   └── .env.example
└── README.md
```

## Setup Instructions

### Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file with your LongCat API key:
   ```bash
   cp .env.example .env
   # Edit .env and add your API key:
   # LONGCAT_API_KEY=your_actual_api_key_here
   ```

5. Start the FastAPI server:
   ```bash
   uvicorn main:app --reload --port 8000
   ```

   The backend will run at: `http://localhost:8000`

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

## Notes

- The backend must be running on port 8000 for the frontend to connect properly
- Update the API URL in `App.tsx` if deploying to production
- For production use, configure CORS properly in `main.py` with specific allowed origins
- The terminal styling uses monospace fonts and green (#00ff66) text on black (#000) background

## License

This project is open source and available under the MIT License.

import { useState } from 'react';
import axios from 'axios';
import ChatWindow from './components/ChatWindow';
import ThinkingToggle from './components/ThinkingToggle';
import './App.css';

interface Message {
  role: 'user' | 'assistant';
  content: string;
}

function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [thinking, setThinking] = useState(false);
  const [loading, setLoading] = useState(false);

  const handleSend = async () => {
    if (!input.trim() || loading) return;

    const userMessage = input.trim();
    setInput('');
    setMessages(prev => [...prev, { role: 'user', content: userMessage }]);
    setLoading(true);

    try {
      const response = await axios.post('http://localhost:5000/chat', {
        message: userMessage,
        thinking: thinking
      });

      setMessages(prev => [...prev, { role: 'assistant', content: response.data.reply }]);
    } catch (error) {
      const errorMsg = axios.isAxiosError(error) 
        ? `Error: ${error.response?.data?.detail || error.message}`
        : 'An unexpected error occurred';
      setMessages(prev => [...prev, { role: 'assistant', content: errorMsg }]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div style={{
      width: '100vw',
      height: '100vh',
      backgroundColor: '#000',
      display: 'flex',
      flexDirection: 'column',
      margin: 0,
      padding: 0,
      overflow: 'hidden'
    }}>
      <div style={{
        padding: '20px',
        borderBottom: '1px solid #00ff66',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center'
      }}>
        <h1 style={{
          color: '#00ff66',
          fontFamily: 'monospace',
          fontSize: '20px',
          margin: 0
        }}>
          {'> AI TERMINAL'}
        </h1>
        <ThinkingToggle thinking={thinking} onToggle={setThinking} />
      </div>

      <ChatWindow messages={messages} />

      <div style={{
        padding: '20px',
        borderTop: '1px solid #00ff66',
        display: 'flex',
        gap: '10px',
        alignItems: 'center'
      }}>
        <span style={{
          color: '#00ff66',
          fontFamily: 'monospace',
          fontSize: '14px'
        }}>
          {'>'}
        </span>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={handleKeyPress}
          disabled={loading}
          placeholder="Type your message..."
          style={{
            flex: 1,
            backgroundColor: '#000',
            border: 'none',
            color: '#00ff66',
            fontFamily: 'monospace',
            fontSize: '14px',
            outline: 'none',
            padding: '8px'
          }}
        />
        <button
          onClick={handleSend}
          disabled={loading}
          style={{
            backgroundColor: '#000',
            border: '1px solid #00ff66',
            color: '#00ff66',
            fontFamily: 'monospace',
            fontSize: '14px',
            padding: '8px 16px',
            cursor: loading ? 'not-allowed' : 'pointer',
            opacity: loading ? 0.5 : 1
          }}
        >
          {loading ? 'SENDING...' : 'SEND'}
        </button>
      </div>
    </div>
  );
}

export default App;

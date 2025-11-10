import { useState, useEffect } from 'react';
import axios from 'axios';
import ChatWindow from './components/ChatWindow';
import ThinkingToggle from './components/ThinkingToggle';
import './App.css';

interface Message {
  role: 'user' | 'assistant';
  content: string;
}

interface DbMessage {
  role: string;
  content: string;
  timestamp?: string;
  thinking?: boolean;
  model?: string;
}

function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [thinking, setThinking] = useState(false);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const fetchChatHistory = async () => {
      try {
        const response = await axios.get<{ messages: DbMessage[] }>('http://localhost:5000/messages');
        if (response.data.messages) {
          const formattedMessages: Message[] = response.data.messages.map((msg) => ({
            role: msg.role as 'user' | 'assistant',
            content: msg.content
          }));
          setMessages(formattedMessages);
        }
      } catch (error) {
        console.error('Error fetching chat history:', error);
      }
    };

    fetchChatHistory();
  }, []);

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
    <div
      style={{
        color: '#fff',
        fontFamily: 'monospace',
        fontSize: '16px',
        height: '100vh',
        backgroundColor: '#111',
        display: 'flex',
        flexDirection: 'column',
        paddingBottom: '20px',
        margin: 0,
        overflow: 'hidden',
      }}
    >
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          padding: '10px 0',
        }}
      >
      
      
      </div>

      <ChatWindow messages={messages} />

      <div
        style={{
          display: 'flex',
          marginTop: '5px',
          padding: '0 0',
          alignItems: 'center',
          gap: '8px',
        }}
      >
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={handleKeyPress}
          disabled={loading}
          placeholder="Type your message..."
          style={{
            flex: 1,
            padding: '8px',
            background: '#111',
            color: '#fff',
            border: '1px solid #333',
            outline: 'none',
          }}
        />
          
        <button
          onClick={handleSend}
          disabled={loading}
          style={{
            padding: '8px 12px',
            background: '#0f0',
            border: 'none',
            borderRadius: '5px',
            cursor: loading ? 'not-allowed' : 'pointer',
            opacity: loading ? 0.6 : 1,
            color: '#111',
            fontWeight: 600,
          }}
        >
          
          {loading ? 'SENDING...' : 'Send'}
        </button>
        <ThinkingToggle thinking={thinking} onToggle={setThinking} />
      </div>
    </div>
  );
}

export default App;
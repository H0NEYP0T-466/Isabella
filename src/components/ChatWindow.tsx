import React, { useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  audioFile?: string;
}

interface ChatWindowProps {
  messages: Message[];
}

const ChatWindow: React.FC<ChatWindowProps> = ({ messages }) => {
  const chatEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  return (
    <div
      style={{
        flex: 1,
        overflowY: 'auto',
      }}
    >
      {messages.length === 0 ? null : (
        messages.map((msg, idx) => (
          <div key={idx} style={{ marginBottom: '8px' }}>
            <span style={{ color: '#0f0' }}>
              {msg.role === 'user' ? '~honeypot' : 'Isabella'}:
            </span>{' '}
            <ReactMarkdown
              children={msg.content}
              components={{
                h1: ({ ...props }) => (
                  <h1 style={{ color: '#ff4040', display: 'inline' }} {...props} />
                ),
                h2: ({ ...props }) => (
                  <h2 style={{ color: '#ff4040', display: 'inline' }} {...props} />
                ),
                h3: ({ ...props }) => (
                  <h3 style={{ color: '#ff4040', display: 'inline' }} {...props} />
                ),
                h4: ({ ...props }) => (
                  <h4 style={{ color: '#ff4040', display: 'inline' }} {...props} />
                ),
                h5: ({ ...props }) => (
                  <h5 style={{ color: '#ff4040', display: 'inline' }} {...props} />
                ),
                h6: ({ ...props }) => (
                  <h6 style={{ color: '#ff4040', display: 'inline' }} {...props} />
                ),
                p: ({ ...props }) => (
                  <p style={{ display: 'inline', color: '#fff' }} {...props} />
                ),
                strong: ({ ...props }) => (
                  <strong style={{ color: '#ff4040' }} {...props} />
                ),
                em: ({ ...props }) => (
                  <em style={{ color: '#ccc', fontStyle: 'italic' }} {...props} />
                ),
                code: ({ ...props }) => (
                  <code
                    style={{
                      background: '#222',
                      color: '#fff',
                      padding: '2px 4px',
                      borderRadius: '4px',
                    }}
                    {...props}
                  />
                ),
              }}
            />
            {msg.audioFile && (
              <div style={{ marginTop: '4px' }}>
                <audio
                  controls
                  src={`http://localhost:5000/tts/audio/${msg.audioFile}`}
                  style={{
                    width: '200px',
                    height: '30px',
                  }}
                />
              </div>
            )}
          </div>
        ))
      )}
      {/* Invisible element to scroll to */}
      <div ref={chatEndRef} />
    </div>
  );
};

export default ChatWindow;
import React from 'react';

interface Message {
  role: 'user' | 'assistant';
  content: string;
}

interface ChatWindowProps {
  messages: Message[];
}

const ChatWindow: React.FC<ChatWindowProps> = ({ messages }) => {
  return (
    <div style={{
      flex: 1,
      overflowY: 'auto',
      padding: '20px',
      fontFamily: 'monospace',
      fontSize: '14px',
      lineHeight: '1.6'
    }}>
      {messages.length === 0 ? (
        <div style={{ color: '#00ff66', opacity: 0.5 }}>
          {'> System ready. Type your message below...'}
        </div>
      ) : (
        messages.map((msg, idx) => (
          <div key={idx} style={{ marginBottom: '15px' }}>
            <div style={{ color: '#00ff66', marginBottom: '5px' }}>
              {msg.role === 'user' ? '> USER:' : '> AI:'}
            </div>
            <div style={{ color: '#00ff66', paddingLeft: '10px', whiteSpace: 'pre-wrap' }}>
              {msg.content}
            </div>
          </div>
        ))
      )}
    </div>
  );
};

export default ChatWindow;

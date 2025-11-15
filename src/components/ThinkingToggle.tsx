import React from 'react';

interface ThinkingToggleProps {
  thinking: boolean;
  onToggle: (thinking: boolean) => void;
}

const ThinkingToggle: React.FC<ThinkingToggleProps> = ({ thinking, onToggle }) => {
  return (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      gap: '10px',
      marginBottom: '10px',
      fontFamily: 'monospace',
      fontSize: '14px'
    }}>
      <label style={{ color: '#00ff66', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '8px' }}>
        <input
          type="checkbox"
          checked={thinking}
          onChange={(e) => onToggle(e.target.checked)}
          style={{
            cursor: 'pointer',
            width: '18px',
            height: '18px'
          }}
        />
        <span>Thinking</span>
      </label>
      <span style={{ color: '#00ff66', opacity: 0.7 }}>
        [{thinking ? 'LongCat-Think' : 'LongCat-Flash-Chat'}]
      </span>
    </div>
  );
};

export default ThinkingToggle;

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
      gap: '12px',
      fontFamily: 'monospace',
      fontSize: '15px',
      padding: '8px 12px',
      borderRadius: '6px',
      backgroundColor: 'rgba(0, 255, 102, 0.05)',
      border: '1px solid rgba(0, 255, 102, 0.2)',
    }}>
      <label style={{ 
        color: '#00ff66', 
        cursor: 'pointer', 
        display: 'flex', 
        alignItems: 'center', 
        gap: '8px',
        fontWeight: '500',
      }}>
        <input
          type="checkbox"
          checked={thinking}
          onChange={(e) => onToggle(e.target.checked)}
          style={{
            cursor: 'pointer',
            width: '18px',
            height: '18px',
            accentColor: '#00ff66',
          }}
        />
        <span>Thinking Mode</span>
      </label>
      <span style={{ color: '#00ff66', opacity: 0.7, fontSize: '13px' }}>
        {thinking ? 'LongCat-Think' : 'LongCat-Flash'}
      </span>
    </div>
  );
};

export default ThinkingToggle;

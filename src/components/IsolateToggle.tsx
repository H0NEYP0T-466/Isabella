import React from 'react';

interface IsolateToggleProps {
  isolate: boolean;
  onToggle: (isolate: boolean) => void;
}

const IsolateToggle: React.FC<IsolateToggleProps> = ({ isolate, onToggle }) => {
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
          checked={isolate}
          onChange={(e) => onToggle(e.target.checked)}
          style={{
            cursor: 'pointer',
            width: '18px',
            height: '18px',
            accentColor: '#00ff66',
          }}
        />
        <span>Isolate Message</span>
      </label>
      <span style={{ color: '#00ff66', opacity: 0.7, fontSize: '13px' }}>
        {isolate ? 'No Context' : 'With Context'}
      </span>
    </div>
  );
};

export default IsolateToggle;

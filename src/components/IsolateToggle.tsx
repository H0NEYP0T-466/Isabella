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
      gap: '10px',
      fontFamily: 'monospace',
      fontSize: '14px'
    }}>
      <label style={{ color: '#00ff66', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '8px' }}>
        <input
          type="checkbox"
          checked={isolate}
          onChange={(e) => onToggle(e.target.checked)}
          style={{
            cursor: 'pointer',
            width: '18px',
            height: '18px'
          }}
        />
        <span>Isolate Message</span>
      </label>
      <span style={{ color: '#00ff66', opacity: 0.7 }}>
        [{isolate ? 'No Context' : 'With Context'}]
      </span>
    </div>
  );
};

export default IsolateToggle;

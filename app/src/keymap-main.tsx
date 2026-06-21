import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { KeymapApp } from './keymap/KeymapApp';

const root = document.getElementById('root');
if (!root) throw new Error('missing #root element');
createRoot(root).render(
  <StrictMode>
    <KeymapApp />
  </StrictMode>,
);

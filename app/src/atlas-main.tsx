import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { AtlasApp } from './atlas/AtlasApp';
import './atlas/atlas.css';

const root = document.getElementById('root');
if (!root) throw new Error('missing #root element');
createRoot(root).render(
  <StrictMode>
    <AtlasApp />
  </StrictMode>,
);

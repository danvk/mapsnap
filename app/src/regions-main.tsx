import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { RegionsApp } from './regions/RegionsApp';

const root = document.getElementById('root');
if (!root) throw new Error('missing #root element');
createRoot(root).render(
  <StrictMode>
    <RegionsApp />
  </StrictMode>,
);

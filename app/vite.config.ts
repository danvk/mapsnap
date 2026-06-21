/// <reference types="vitest/config" />
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  base: '/mapsnap/',
  plugins: [react()],
  server: {
    proxy: {
      '/iiif': 'http://localhost:8182',
    },
  },
  test: {
    environment: 'jsdom',
  },
});

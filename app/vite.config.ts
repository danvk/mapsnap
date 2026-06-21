/// <reference types="vitest/config" />
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  base: '/mapsnap/',
  plugins: [react()],
  build: {
    rollupOptions: {
      input: {
        main: 'index.html',
        keymap: 'keymap.html',
      },
    },
  },
  server: {
    proxy: {
      '/iiif': 'http://localhost:8182',
      '/api': 'http://localhost:8183',
    },
  },
  test: {
    environment: 'jsdom',
  },
});

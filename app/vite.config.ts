import { defineConfig } from 'vite';

export default defineConfig({
  base: '/mapsnap/',
  server: {
    proxy: {
      '/iiif': 'http://localhost:8182',
    },
  },
});

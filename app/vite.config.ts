/// <reference types="vitest/config" />
import { defineConfig, type Plugin } from 'vite';
import react from '@vitejs/plugin-react';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const repoRoot = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  '..',
);
const dataDir = path.join(repoRoot, 'data');

// Content types for the file kinds found under data/.
const MIME_BY_EXT: Record<string, string> = {
  '.json': 'application/json',
  '.geojson': 'application/geo+json',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.png': 'image/png',
  '.webp': 'image/webp',
  '.gif': 'image/gif',
  '.tif': 'image/tiff',
  '.tiff': 'image/tiff',
  '.txt': 'text/plain',
  '.csv': 'text/csv',
};

// Guess a Content-Type from a file path's extension.
function contentTypeFor(filePath: string): string {
  return (
    MIME_BY_EXT[path.extname(filePath).toLowerCase()] ??
    'application/octet-stream'
  );
}

/**
 * Serve the repo-root `data/` directory over the dev server.
 *
 * Vite's root is `app/`, so `data/` (a sibling) is otherwise unreachable. This
 * exposes it at `<base>data/...` (e.g. `/mapsnap/data/streets.json`) so test
 * data can be loaded by URL via the app's `?files=` deep link.
 */
function serveDataDir(): Plugin {
  return {
    name: 'serve-data-dir',
    configureServer(server) {
      const base = server.config.base; // e.g. '/mapsnap/'
      server.middlewares.use((req, res, next) => {
        if (!req.url) return next();

        // Custom middleware runs before Vite strips the base, but be tolerant of
        // either form by stripping the base if present, else the leading slash.
        let pathname = decodeURIComponent(
          new URL(req.url, 'http://localhost').pathname,
        );
        pathname = pathname.startsWith(base)
          ? pathname.slice(base.length)
          : pathname.replace(/^\//, '');
        if (!pathname.startsWith('data/')) return next();

        const filePath = path.join(dataDir, pathname.slice('data/'.length));
        // Refuse paths that escape data/ via `..` or symlinks.
        if (filePath !== dataDir && !filePath.startsWith(dataDir + path.sep)) {
          res.statusCode = 403;
          return res.end('Forbidden');
        }

        fs.stat(filePath, (err, stat) => {
          if (err || !stat.isFile()) return next();
          res.setHeader('Content-Type', contentTypeFor(filePath));
          fs.createReadStream(filePath).pipe(res);
        });
      });
    },
  };
}

export default defineConfig({
  base: '/mapsnap/',
  plugins: [react(), serveDataDir()],
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
      '/iiif-api': 'http://localhost:8182',
      '/iiif': 'http://localhost:8182',
      '/api': 'http://localhost:8182',
      '/notes-api': 'http://localhost:8182',
    },
  },
  test: {
    environment: 'jsdom',
  },
});

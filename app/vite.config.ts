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
          if (err || !stat.isFile()) {
            // A missing data file is a 404, not the app. Falling through sent
            // index.html with a 200, so a typo'd or stale data URL looked like
            // a successful fetch of HTML (#241).
            res.statusCode = 404;
            return res.end(`Not found: ${pathname}`);
          }
          res.setHeader('Content-Type', contentTypeFor(filePath));
          fs.createReadStream(filePath).pipe(res);
        });
      });
    },
  };
}

/**
 * 404 for file-looking URLs nothing served, instead of the SPA fallback.
 *
 * Vite answers any unmatched path with index.html and a 200, which is right
 * for an app that routes on the path -- this one routes on the query string,
 * so every extension-bearing URL that reaches the fallback is a mistake, and
 * silently returning HTML makes it look like a successful fetch (#241).
 *
 * Runs BEFORE Vite's own middlewares (a post hook would sit after the fallback
 * and never see these), so it must skip anything Vite legitimately serves:
 * its internal endpoints, and real files under the app root or public/.
 */
function notFoundForMissingFiles(): Plugin {
  return {
    name: 'not-found-for-missing-files',
    configureServer(server) {
      const base = server.config.base;
      const appRoot = server.config.root;
      const publicDir = server.config.publicDir;
      server.middlewares.use((req, res, next) => {
        if (!req.url) return next();
        const pathname = decodeURIComponent(
          new URL(req.url, 'http://localhost').pathname,
        );
        const rel = pathname.startsWith(base)
          ? pathname.slice(base.length)
          : pathname.replace(/^\//, '');
        // Extensionless paths are app entry points; let the fallback have them.
        if (!path.extname(rel)) return next();
        // Vite internals: /@vite/client, /@fs/..., /@id/..., node_modules, and
        // the source tree it compiles on the fly.
        if (/^(@|node_modules\/|src\/|\.vite\/)/.test(rel)) return next();
        // data/ has its own handler above, which 404s on its own.
        if (rel.startsWith('data/')) return next();
        for (const dir of [appRoot, publicDir]) {
          if (!dir) continue;
          const candidate = path.join(dir, rel);
          if (
            (candidate === dir || candidate.startsWith(dir + path.sep)) &&
            fs.existsSync(candidate) &&
            fs.statSync(candidate).isFile()
          ) {
            return next();
          }
        }
        res.statusCode = 404;
        res.end(`Not found: ${pathname}`);
      });
    },
  };
}

export default defineConfig({
  base: '/mapsnap/',
  plugins: [react(), serveDataDir(), notFoundForMissingFiles()],
  build: {
    rollupOptions: {
      input: {
        main: 'index.html',
        keymap: 'keymap.html',
        adjacency: 'adjacency.html',
        regions: 'regions.html',
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

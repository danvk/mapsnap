/**
 * The atlas alone, as a static page for mapsnap.org.
 *
 *   VITE_ATLAS_DATA_URL=https://data.mapsnap.org/atlas/v2.0 \
 *     npx vite build --config vite.atlas.config.ts
 *
 * writes dist-atlas/atlas.html and dist-atlas/atlas-assets/, which go at the
 * root of the mapsnap.org site, so the atlas is served at /atlas beside the
 * item page at /item. Nothing in public/ is copied: the atlas's data is
 * uploaded to data.mapsnap.org separately (docs/corpus-run.md), and the page
 * reads it from VITE_ATLAS_DATA_URL.
 */
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  base: '/',
  plugins: [react()],
  publicDir: false,
  build: {
    outDir: 'dist-atlas',
    emptyOutDir: true,
    // Its own directory, so the files can't collide with the rest of the site.
    assetsDir: 'atlas-assets',
    rollupOptions: { input: { atlas: 'atlas.html' } },
  },
});

/**
 * Fetch every URL the viewer's own libraries would request, and report misses.
 *
 *   node app/validate-local-iiif.mjs ~/.cache/mapsnap/local-iiif/annotations/*.json
 *
 * It lives under app/ because that is where the @allmaps packages are, and
 * node resolves bare imports from the script's own directory. The URLs come from
 * `image.getTileImageRequest`, the function the renderer itself calls -- an
 * earlier version of this reimplemented the tile arithmetic and agreed with
 * itself while disagreeing with the viewer by one pixel, which is a 404.
 */
import { readFileSync } from 'fs';
import { IIIF } from '@allmaps/iiif-parser';
import { parseAnnotation } from '@allmaps/annotation';

let ok = 0;
const misses = [];
for (const path of process.argv.slice(2)) {
  const maps = parseAnnotation(JSON.parse(readFileSync(path, 'utf8')));
  for (const map of maps) {
    const info = await (await fetch(`${map.resource.id}/info.json`)).json();
    const image = IIIF.parse(info);
    const urls = [image.getImageUrl({})];
    for (const zoom of image.tileZoomLevels) {
      for (let column = 0; column < zoom.columns; column++) {
        for (let row = 0; row < zoom.rows; row++) {
          urls.push(image.getImageUrl(image.getTileImageRequest(zoom, column, row)));
        }
      }
    }
    for (const size of image.sizes ?? []) {
      urls.push(`${image.uri}/full/${size.width},${size.height}/0/default.jpg`);
    }
    for (const url of urls) {
      const response = await fetch(url);
      // The body has to be consumed even when it is not wanted: leaving it
      // unread makes undici destroy the socket, which surfaces as ECONNRESET
      // or EPIPE on a later request rather than on this one.
      const bytes = (await response.arrayBuffer()).byteLength;
      if (response.ok && bytes > 0) {
        ok++;
      } else {
        misses.push(`${response.status} ${bytes}B ${url}`);
      }
    }
  }
}
console.log(`  ${ok} URLs resolve, ${misses.length} missing`);
for (const miss of misses.slice(0, 10)) console.log(`    ${miss}`);
process.exit(misses.length ? 1 : 0);
